import { ArrowUpRight, ArrowDownRight, Minus, ArrowRight } from 'lucide-react';
import './RecommendationCard.css';

interface Props {
  asset: { label: string; price: number; signal: string; action?: number; probs?: number[]; model_eval_time?: string; quote_time?: string };
  onExplain: () => void;
}

export default function RecommendationCard({ asset, onExplain }: Props) {
  const signal = asset.signal.toUpperCase();
  const inferred = /BUY|LONG/.test(signal) ? 1 : /SELL|SHORT/.test(signal) ? 2 : /NEUTRAL|CASH|HOLD/.test(signal) ? 0 : -1;
  const action = asset.action !== undefined && [0, 1, 2].includes(asset.action) ? asset.action : inferred;
  const options = [
    { label: 'Cash', title: 'Cash target', description: 'At the evaluation timestamp, the saved model targeted cash for this asset.', icon: Minus, color: '#93b4ff' },
    { label: 'Long', title: 'Long target', description: 'At the evaluation timestamp, the saved model targeted a long position. This historical signal is not a current trade instruction.', icon: ArrowUpRight, color: '#6ee7b7' },
    { label: 'Short', title: 'Short target', description: 'At the evaluation timestamp, the saved model targeted a short position. This historical signal is not a current trade instruction.', icon: ArrowDownRight, color: '#fda4af' },
  ];
  const target = options[action];
  const Icon = target?.icon ?? Minus;
  const probs = asset.probs;
  const valid = probs?.length === 3 && probs.every(p => Number.isFinite(p) && p >= 0 && p <= 1)
    && Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < .02;
  return (
    <section className="recommendation-card" aria-label="Saved model target">
      <div className="recommendation-main">
        <div className="recommendation-eyebrow"><span className="recommendation-dot" /> MODEL SNAPSHOT</div>
        <div className="recommendation-heading">
          <span className="recommendation-icon" style={{ color: target?.color }}><Icon size={25} /></span>
          <h3>{target?.title ?? 'Signal unavailable'}</h3>
        </div>
        <p>{target?.description ?? 'A model signal is not available for this asset yet.'}</p>
        <button className="recommendation-explain" onClick={onExplain}>Why this signal? <ArrowRight size={15} /></button>
      </div>
      <div className="recommendation-evidence">
        <div className="recommendation-price"><span>{asset.label} <small>· Latest available price</small></span>
          <strong>{Number.isFinite(asset.price) && asset.price > 0 ? asset.price.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: asset.price < 1 ? 5 : 2 }) : 'Unavailable'}</strong>
        </div>
        <div className="recommendation-prob-heading">Model probabilities <span>Cash / Long / Short</span></div>
        {valid ? options.map((option, i) => (
          <div className="recommendation-prob" key={option.label}>
            <span>{option.label}{action === i && <small>Selected</small>}</span>
            <div className="recommendation-track" role="meter" aria-label={`${option.label} probability`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Number((probs[i] * 100).toFixed(1))}>
              <div style={{ width: `${probs[i] * 100}%`, background: option.color }} />
            </div><strong>{(probs[i] * 100).toFixed(1)}%</strong>
          </div>
        )) : <p className="recommendation-missing">Action probabilities unavailable.</p>}
        <p className="recommendation-note">Action probabilities describe the policy's choices, not the chance of profit.</p>
      </div>
      <div className="recommendation-footer">
        <span>Quote timestamp <strong>{asset.quote_time || 'Unavailable'}</strong></span>
        <span>Model evaluated <strong>{asset.model_eval_time || 'Unavailable'}</strong></span>
      </div>
    </section>
  );
}
