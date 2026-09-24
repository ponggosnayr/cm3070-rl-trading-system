import shap
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_shap_explainer(model, background_data, vec_normalize=None):
    """
    Initialize a SHAP KernelExplainer for a Stable Baselines 3 model.
    SHAP requires a function that takes a numpy array and returns probabilities/logits.
    """
    
    def predict_logits(obs_numpy):
        # Apply VecNormalize normalization if wrapper is provided
        if vec_normalize is not None and hasattr(vec_normalize, "normalize_obs"):
            obs_to_predict = vec_normalize.normalize_obs(obs_numpy)
        else:
            obs_to_predict = obs_numpy

        # Convert numpy back to torch tensor and move to the same device as the model
        obs_tensor = torch.as_tensor(obs_to_predict).float().to(model.device)
        
        # SB3 uses model.policy to handle the forward pass
        # We want the 'logits' (action scores) from the actor network
        with torch.no_grad():
            # Get the features using the extractor
            features = model.policy.extract_features(obs_tensor)
            if model.policy.share_features_extractor:
                # Pass through the mlp_extractor to get latent representation for policy (latent_pi)
                latent_pi, _ = model.policy.mlp_extractor(features)
            else:
                # When not sharing, features is a tuple: (pi_features, vf_features)
                pi_features, _ = features
                latent_pi = model.policy.mlp_extractor.forward_actor(pi_features)
            # Pass through the action net to get raw logits
            logits = model.policy.action_net(latent_pi)
            
        return logits.cpu().numpy()

    # KernelExplainer is model-agnostic and robust for complex SB3 policies
    explainer = shap.KernelExplainer(predict_logits, background_data)
    return explainer

def compute_shap_values(model, current_obs, background_data, nsamples=100, vec_normalize=None):
    """
    Compute SHAP values for a single observation.
    current_obs: numpy array (frame-stacked)
    """
    explainer = get_shap_explainer(model, background_data, vec_normalize=vec_normalize)
    shap_values = explainer.shap_values(current_obs, nsamples=nsamples, l1_reg="num_features(10)")
    return shap_values

def plot_shap_summary(shap_values, feature_names, action_names, target_action_idx):
    """
    Plot a bar chart of SHAP values for a specific action.
    shap_values: Array of shape (1, 80, 5) or list of arrays depending on SHAP version.
    """
    # Handle both SHAP v0.45 (list of arrays) and v0.51+ (single 3D array)
    if isinstance(shap_values, list):
        vals = shap_values[target_action_idx][0]
    else:
        # Array shape is (N_samples, N_features, N_classes) -> (1, 80, 5)
        vals = shap_values[0, :, target_action_idx]
    
    # Fix #3: The actual observation has 11 features per step × 8 stacked steps = 88 total.
    # The comment previously said "80 features (10 x 8)" which was incorrect and would cause
    num_features = len(feature_names)
    agg_vals = np.zeros(num_features)
    for i in range(num_features):
        # Sum SHAP values across all 8 time steps for this feature
        agg_vals[i] = np.sum(vals[i::num_features])
        
    # Sort features by importance
    indices = np.argsort(np.abs(agg_vals))
    sorted_features = [feature_names[i] for i in indices]
    sorted_vals = agg_vals[indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#EF4444' if v < 0 else '#10B981' for v in sorted_vals]
    ax.barh(sorted_features, sorted_vals, color=colors)
    ax.set_title(f"Feature Importance for Action: {action_names[target_action_idx]}", color='white', fontsize=12)
    ax.set_xlabel("SHAP Value (Impact on Decision)", color='#94A3B8')
    
    # Styling for dark theme
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')
    ax.spines['bottom'].set_color('#334155')
    ax.spines['top'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['right'].set_color('#334155')
    ax.tick_params(axis='x', colors='#94A3B8')
    ax.tick_params(axis='y', colors='#94A3B8')
    ax.grid(axis='x', linestyle='--', alpha=0.1)
    
    return fig

def generate_global_model_persona(model, background_data, feature_names, sample_size=500):
    """
    Analyzes the model's global strategy by averaging SHAP values across hundreds of states.
    Returns a text description of the model's 'Persona'.
    """
    explainer = get_shap_explainer(model, background_data)
    
    # Take a random sample of historical market states to test the model on
    sample_size = min(sample_size, background_data.shape[0])
    indices = np.random.choice(background_data.shape[0], sample_size, replace=False)
    test_states = background_data[indices]
    
    # Calculate SHAP values for all sampled states
    # This might take a minute depending on sample size
    shap_values = explainer.shap_values(test_states, nsamples=200)
    
    # We want to look at the overall magnitude of importance (absolute values)
    # Average across all samples, and sum across the 8 time steps
    if isinstance(shap_values, list):
        # Taking the absolute mean across all samples for the 'Buy' action (index 1)
        global_importance = np.mean(np.abs(shap_values[1]), axis=0) 
    else:
        # SHAP 0.51+
        global_importance = np.mean(np.abs(shap_values[:, :, 1]), axis=0)
        
    num_features = len(feature_names)
    agg_vals = np.zeros(num_features)
    for i in range(num_features):
        agg_vals[i] = np.sum(global_importance[i::num_features])
        
    # Sort features by highest global importance
    top_indices = np.argsort(agg_vals)[::-1]
    
    # Generate a simple text summary
    primary = feature_names[top_indices[0]]
    secondary = feature_names[top_indices[1]]
    
    persona = f"This model is primarily a **{primary}**-driven trader, heavily supported by **{secondary}**."
    
    if "RSI" in primary or "RSI" in secondary:
        persona += " It is highly sensitive to momentum and looks for overbought/oversold extremes."
    if "MACD" in primary or "MACD" in secondary:
        persona += " It acts as a trend-follower, waiting for momentum confirmation before entering."
    if "Volume" in primary or "Volume" in secondary:
        persona += " It requires high market participation to validate its trades, ignoring low-volume noise."
        
    return persona, top_indices, agg_vals

def generate_regime_shap_summary(model, background_data, regime_labels, feature_names, sample_size=500):
    """
    Analyzes the model's feature importance separated by market regimes.
    """
    explainer = get_shap_explainer(model, background_data)
    
    sample_size = min(sample_size, background_data.shape[0])
    indices = np.random.choice(background_data.shape[0], sample_size, replace=False)
    test_states = background_data[indices]
    test_regimes = regime_labels[indices]
    
    shap_values = explainer.shap_values(test_states, nsamples=100)
    
    if isinstance(shap_values, list):
        importance = np.abs(shap_values[1])
    else:
        importance = np.abs(shap_values[:, :, 1])
        
    num_features = len(feature_names)
    regime_results = {}
    
    unique_regimes = np.unique(test_regimes)
    for r in unique_regimes:
        r_indices = np.where(test_regimes == r)[0]
        if len(r_indices) == 0:
            continue
            
        r_importance = np.mean(importance[r_indices], axis=0)
        agg_vals = np.zeros(num_features)
        for i in range(num_features):
            agg_vals[i] = np.sum(r_importance[i::num_features])
            
        top_indices = np.argsort(agg_vals)[::-1]
        regime_results[r] = {
            "top_features": [feature_names[i] for i in top_indices[:5]],
            "top_values": [agg_vals[i] for i in top_indices[:5]],
            "all_vals": agg_vals
        }
        
    return regime_results
