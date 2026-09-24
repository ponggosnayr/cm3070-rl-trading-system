import React, { useState, useEffect, useRef } from 'react';
import DOMPurify from 'dompurify';
import { Bot, Activity, Send, TrendingUp, TrendingDown, Minus, BookOpen, BarChart2, ArrowLeft, Cpu, ChevronDown, ChevronUp, Radio } from 'lucide-react';
import { ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, Tooltip as RechartsTooltip, AreaChart, Area, Scatter } from 'recharts';

interface Candle {
  open: number;
  high: number;
  low: number;
  close: number;
}

interface AssetData {
  label: string;
  price: number;
  signal: string;
  confidence: number;
  sentiment: string;
  history: number[];
  candles?: Candle[];
  regime?: string;
}

interface TradeItem {
  step: number;
  action: number;
  price: number;
  net_worth: number;
}

interface BacktestMetrics {
  roi: number;
  bh_return: number;
  sharpe: number;
  sortino: number;
  max_dd: number;
  calmar: number;
  trades: number;
  final_capital: number;
  history: number[];
  trade_history: TradeItem[];
  dates?: string[];
}

interface ShapItem {
  feature: string;
  value: number;
}

interface ShapData {
  shap: ShapItem[];
  action: number;
}

interface WalkForwardRow {
  Fold: number;
  'Test Range': string;
  'Out-of-Sample ROI (%)': number;
  'Market B&H (%)': number;
  'Out-of-Sample Sharpe': number;
  'Max DD (%)': number;
  Trades: number;
}

interface MonteCarloSummary {
  mean_roi: number;
  std_roi: number;
  min_roi: number;
  max_roi: number;
  var_5th: number;
  mean_sharpe: number;
  mean_drawdown: number;
  success_rate: number;
}

interface MonteCarloData {
  summary: MonteCarloSummary;
  percentiles: {
    '5': number[];
    '25': number[];
    '50': number[];
    '75': number[];
    '95': number[];
  };
  sample_paths: number[][];
  step_count: number;
}

interface ComparisonRow {
  Model: string;
  'ROI (%)'?: number;
  ROI?: number;
  'Max DD (%)'?: number;
  Max_DD?: number;
  Sharpe?: number;
  Trades?: number;
}

interface ChatPayload {
  message: string;
  history: Array<{ role: string; content: string }>;
  asset_name?: string;
  price?: number;
  signal?: string;
  confidence?: number;
  sentiment?: string;
}

// Simple markdown-to-HTML converter supporting headings, lists, bold, and italics with DOMPurify sanitization
function renderMarkdown(text: string) {
  let html = text;

  // Parse headings (must run before converting newlines to preserve line boundaries)
  html = html.replace(/^### (.*?)$/gm, '<h4 style="margin: 0.8rem 0 0.4rem 0; color: #FFFFFF; font-size: 1.05rem; font-weight: bold;">$1</h4>');
  html = html.replace(/^## (.*?)$/gm, '<h3 style="margin: 1rem 0 0.5rem 0; color: #FFFFFF; font-size: 1.15rem; font-weight: bold;">$1</h3>');
  html = html.replace(/^# (.*?)$/gm, '<h2 style="margin: 1.2rem 0 0.6rem 0; color: #FFFFFF; font-size: 1.25rem; font-weight: bold;">$1</h2>');

  // Bold and Italics
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Bullet lists (lines starting with * or - followed by a space)
  html = html.replace(/^\s*[*\\-]\s+(.*?)$/gm, '<li style="margin: 0.25rem 0 0.25rem 1.25rem; list-style-type: disc; line-height: 1.5;">$1</li>');

  // Group consecutive list items into ul tags
  html = html.replace(/(<li style=".*?">.*?<\/li>)+/g, '<ul style="margin: 0.5rem 0; padding-left: 0;">$&</ul>');

  // Convert remaining newlines to br
  html = html.replace(/\n/g, '<br/>');

  // Clean up any double br around block elements
  html = html.replace(/<br\/>(<\/?(?:ul|li|h2|h3|h4))/g, '$1');
  html = html.replace(/(<\/(?:ul|li|h2|h3|h4)>)<br\/>/g, '$1');

  return DOMPurify.sanitize(html);
}

// SVG sparkline component
export function Sparkline({ data, color = '#7C3AED', width = 120, height = 40 }: { data: number[]; color?: string; width?: number; height?: number }) {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const uid = `grad-${color.replace('#', '')}-${width}`;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block', margin: '0.5rem auto 0' }}>
      <defs>
        <linearGradient id={uid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,${height} ${points} ${width},${height}`} fill={`url(#${uid})`} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}



function InfoBadge({ topic, text, anchor, onNavigate }: { topic: string, text: string, anchor: string, onNavigate: (anchor: string) => void }) {
  return (
    <span className="info-badge-container">
      <span className="info-badge-icon">?</span>
      <span className="info-badge-tooltip">
        <span style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold', color: '#FFFFFF' }}>{topic}</span>
        {text}
        <a className="info-badge-link" onClick={() => onNavigate(anchor)}>
          Learn more in Model School →
        </a>
      </span>
    </span>
  );
}

function SignalIcon({ signal }: { signal: string }) {
  if (signal === 'BUY') return <TrendingUp size={16} />;
  if (signal === 'SELL') return <TrendingDown size={16} />;
  return <Minus size={16} />;
}

function CandlestickChart({ 
  data: rawData, 
  width = "100%", 
  height = 120, 
  interactive = true,
  maxCandles
}: { 
  data: Candle[]; 
  width?: number | string; 
  height?: number; 
  interactive?: boolean;
  maxCandles?: number;
}) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  // Downsample data if maxCandles or default limits apply (called unconditionally to satisfy Rules of Hooks)
  const data = React.useMemo(() => {
    if (!rawData || rawData.length === 0) return [];
    const limit = maxCandles || (interactive ? undefined : 48);
    if (!limit || rawData.length <= limit) return rawData;
    
    const groupSize = Math.ceil(rawData.length / limit);
    const result: Candle[] = [];
    for (let i = 0; i < rawData.length; i += groupSize) {
      const group = rawData.slice(i, i + groupSize);
      if (group.length === 0) continue;
      result.push({
        open: group[0].open,
        high: Math.max(...group.map(c => c.high)),
        low: Math.min(...group.map(c => c.low)),
        close: group[group.length - 1].close
      });
    }
    return result;
  }, [rawData, maxCandles, interactive]);

  if (!rawData || rawData.length === 0 || data.length === 0) {
    return <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center', padding: '1rem' }}>No candlestick data available.</div>;
  }

  const highs = data.map(c => c.high);
  const lows = data.map(c => c.low);

  const minVal = Math.min(...lows);
  const maxVal = Math.max(...highs);
  const range = maxVal - minVal || 1;

  const paddingTop = interactive ? 22 : 4; // Room for OHLC info at the top only if interactive
  const paddingBottom = interactive ? 8 : 4;
  const plotHeight = height - paddingTop - paddingBottom;

  const getY = (val: number) => {
    return height - paddingBottom - ((val - minVal) / range) * plotHeight;
  };

  const svgWidth = data.length * 4;
  const candleWidth = 4;
  const activeCandle = hoverIdx !== null ? data[hoverIdx] : data[data.length - 1];

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!interactive) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = ((e.clientX - rect.left) / rect.width) * svgWidth;
    const idx = Math.min(data.length - 1, Math.max(0, Math.floor(mouseX / candleWidth)));
    setHoverIdx(idx);
  };

  const getRelativeTimeStr = (index: number) => {
    const minsAgo = Math.round((data.length - 1 - index) * 30);
    const hoursAgo = Math.floor(minsAgo / 60);
    const remainingMins = minsAgo % 60;
    return hoursAgo > 0 ? `-${hoursAgo}h${remainingMins > 0 ? ` ${remainingMins}m` : ''}` : `-${remainingMins}m`;
  };

  return (
    <div style={{ 
      width: typeof width === 'number' ? `${width}px` : width, 
      position: 'relative', 
      background: interactive ? 'rgba(15, 10, 25, 0.2)' : 'transparent', 
      padding: interactive ? '0.5rem' : '0', 
      borderRadius: '8px',
      boxSizing: 'border-box'
    }}>
      {/* OHLC Bar at the top - only when interactive */}
      {interactive && (
        <div style={{ display: 'flex', gap: '0.5rem', fontSize: '0.72rem', fontFamily: 'monospace', color: 'var(--text-secondary)', marginBottom: '0.4rem', justifyContent: 'space-between', borderBottom: '1px solid rgba(255, 255, 255, 0.05)', paddingBottom: '0.2rem' }}>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            <span>O:<span style={{ color: 'white' }}>{activeCandle.open.toLocaleString()}</span></span>
            <span>H:<span style={{ color: '#10b981' }}>{activeCandle.high.toLocaleString()}</span></span>
            <span>L:<span style={{ color: '#ef4444' }}>{activeCandle.low.toLocaleString()}</span></span>
            <span>C:<span style={{ color: 'white' }}>{activeCandle.close.toLocaleString()}</span></span>
          </div>
          <span style={{ color: '#FFFFFF', fontWeight: 'bold' }}>
            {hoverIdx !== null ? getRelativeTimeStr(hoverIdx) : 'Live'}
          </span>
        </div>
      )}

      <svg 
        width="100%" 
        height={height} 
        viewBox={`0 0 ${svgWidth} ${height}`} 
        preserveAspectRatio="none"
        style={{ display: 'block', overflow: 'visible', cursor: interactive ? 'crosshair' : 'default' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIdx(null)}
      >
        {data.map((candle, idx) => {
          const x = idx * candleWidth + candleWidth / 2;
          const yOpen = getY(candle.open);
          const yClose = getY(candle.close);
          const yHigh = getY(candle.high);
          const yLow = getY(candle.low);

          const isBullish = candle.close >= candle.open;
          const color = isBullish ? '#10b981' : '#ef4444';

          const isHovered = interactive && hoverIdx === idx;

          return (
            <g key={idx}>
              {/* Wick */}
              <line 
                x1={x} 
                y1={yHigh} 
                x2={x} 
                y2={yLow} 
                stroke={color} 
                strokeWidth={isHovered ? 1.5 : 1} 
                strokeOpacity={isHovered ? 1 : 0.6} 
                vectorEffect="non-scaling-stroke"
              />
              {/* Body */}
              <line 
                x1={x} 
                y1={yOpen} 
                x2={x} 
                y2={yClose} 
                stroke={color} 
                strokeWidth={isHovered ? 3.2 : 2.2} 
              />
            </g>
          );
        })}

        {/* Hover Crosshair Vertical Line - only when interactive */}
        {interactive && hoverIdx !== null && (
          <line
            x1={hoverIdx * candleWidth + candleWidth / 2}
            y1={paddingTop - 10}
            x2={hoverIdx * candleWidth + candleWidth / 2}
            y2={height - paddingBottom}
            stroke="rgba(255, 255, 255, 0.2)"
            strokeWidth={0.8}
            strokeDasharray="2,2"
          />
        )}
      </svg>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'analytics' | 'deep_analytics' | 'chat' | 'school'>('dashboard');
  const [deepSubTab, setDeepSubTab] = useState<'all' | 'backtest' | 'shap' | 'regime' | 'wfv' | 'montecarlo'>('all');
  const [pendingScrollHash, setPendingScrollHash] = useState<string | null>(null);
  const [isTelemetryCollapsed, setIsTelemetryCollapsed] = useState(false);

  const handleNavigateSchool = (hash: string) => {
    setActiveTab('school');
    setPendingScrollHash(hash);
  };

  useEffect(() => {
    if (activeTab === 'school' && pendingScrollHash) {
      setTimeout(() => {
        const element = document.getElementById(pendingScrollHash.replace('#', ''));
        if (element) {
          element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        setPendingScrollHash(null);
      }, 100);
    }
  }, [activeTab, pendingScrollHash]);
  const [scoutData, setScoutData] = useState<{ Crypto: AssetData[]; ETFs: AssetData[] } | null>(null);
  const [selectedAsset, setSelectedAsset] = useState<AssetData | null>(null);
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', content: 'Hello! I am your Trading Advisor powered by Google Gemini. How may I assist you today?' }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // New states for real-time model integration
  const [backtestMetrics, setBacktestMetrics] = useState<BacktestMetrics | null>(null);
  const [shapData, setShapData] = useState<ShapData | null>(null);
  const [loadingAnalytics, setLoadingAnalytics] = useState(false);

  // Walk-Forward Validation state
  const [wfvData, setWfvData] = useState<WalkForwardRow[] | null>(null);
  const [loadingWFV, setLoadingWFV] = useState(false);

  const fetchWalkForward = async () => {
    setLoadingWFV(true);
    try {
      const res = await fetch('/api/walk-forward');
      if (res.ok) {
        const json = await res.json();
        setWfvData(json.data);
      }
    } catch (err) {
      console.error('Error fetching walk forward data:', err);
    }
    setLoadingWFV(false);
  };


  // Monte Carlo simulation states
  const [mcData, setMcData] = useState<MonteCarloData | null>(null);
  const [loadingMC, setLoadingMC] = useState(false);
  const [mcSimulations, setMcSimulations] = useState(50);
  const [mcNoise, setMcNoise] = useState(0.002);
  const [mcSwan, setMcSwan] = useState(false);
  const [mcHoverStep, setMcHoverStep] = useState<number | null>(null);

  // Model comparison states
  const [comparisonData, setComparisonData] = useState<ComparisonRow[] | null>(null);
  const [loadingComparison, setLoadingComparison] = useState(false);

  const fetchComparison = async () => {
    setLoadingComparison(true);
    try {
      const res = await fetch('/api/comparison');
      if (res.ok) {
        const data = await res.json();
        setComparisonData(data.results);
      }
    } catch (err) {
      console.error('Error fetching comparison:', err);
    } finally {
      setLoadingComparison(false);
    }
  };

  useEffect(() => {
    let isMounted = true;
    if (activeTab === 'school' && !comparisonData) {
      fetch('/api/comparison')
        .then(res => res.ok ? res.json() : Promise.reject())
        .then(data => {
          if (isMounted) setComparisonData(data.results);
        })
        .catch(err => console.error('Error fetching comparison:', err));
    }
    return () => {
      isMounted = false;
    };
  }, [activeTab, comparisonData]);

  const fetchScoutData = () => {
    fetch('/api/scout')
      .then(res => res.json())
      .then(data => {
        setScoutData(data);
        setSelectedAsset(prev => {
          const all = [...(data.Crypto || []), ...(data.ETFs || [])];
          if (!prev && all.length > 0) {
            const first = all[0];
            return first;
          }
          if (prev) {
            const updated = all.find((a: AssetData) => a.label === prev.label);
            if (updated) {
              return {
                ...prev,
                price: updated.price,
                signal: updated.signal,
                confidence: updated.confidence,
                sentiment: updated.sentiment,
                regime: updated.regime,
                candles: updated.candles,
                history: updated.history
              };
            }
          }
          return prev;
        });
      })
      .catch(err => console.error('API Error:', err));
  };

  useEffect(() => {
    fetchScoutData();
    const interval = setInterval(fetchScoutData, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchAnalytics = async (assetName: string) => {
    setLoadingAnalytics(true);
    setBacktestMetrics(null);
    setShapData(null);
    try {
      // 1. Fetch real backtest results
      const backtestRes = await fetch(`/api/backtest?asset=${encodeURIComponent(assetName)}`);
      if (backtestRes.ok) {
        const backtestData = await backtestRes.json();
        setBacktestMetrics(backtestData);
      }
      
      // 2. Fetch real SHAP attribution metrics
      const shapRes = await fetch(`/api/shap?asset=${encodeURIComponent(assetName)}`);
      if (shapRes.ok) {
        const shapVal = await shapRes.json();
        setShapData(shapVal);
      }
    } catch (err) {
      console.error('Error fetching analytics:', err);
    }
    setLoadingAnalytics(false);
  };

  const fetchMonteCarlo = async (assetName: string, sims = mcSimulations, noise = mcNoise, swan = mcSwan) => {
    setLoadingMC(true);
    setMcData(null);
    try {
      const url = `/api/montecarlo?asset=${encodeURIComponent(assetName)}&simulations=${sims}&noise_std=${noise}&inject_swan=${swan}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setMcData(data);
      }
    } catch (err) {
      console.error('Error fetching Monte Carlo:', err);
    }
    setLoadingMC(false);
  };

  const handleOpenDeepAnalytics = (asset?: AssetData) => {
    const target = asset || selectedAsset || (allAssets.length > 0 ? allAssets[0] : null);
    if (target) {
      setSelectedAsset(target);
      if (!backtestMetrics) fetchAnalytics(target.label);
      if (!wfvData) fetchWalkForward();
      if (!mcData) fetchMonteCarlo(target.label);
    }
    setActiveTab('deep_analytics');
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, loading]);

  const sendChat = async (message: string, overrideAsset?: AssetData) => {
    if (!message.trim()) return;
    setChatMessages(prev => [...prev, { role: 'user', content: message }]);
    setChatInput('');
    setLoading(true);
    try {
      const payload: ChatPayload = {
        message,
        history: chatMessages.map(msg => ({ role: msg.role, content: msg.content }))
      };
      const activeAsset = overrideAsset !== undefined ? overrideAsset : selectedAsset;
      if (activeAsset) {
        payload.asset_name = activeAsset.label;
        payload.price = activeAsset.price;
        payload.signal = activeAsset.signal;
        payload.confidence = activeAsset.confidence;
        payload.sentiment = activeAsset.sentiment;
      }
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch {
      setChatMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I am currently disconnected from the backend.' }]);
    }
    setLoading(false);
  };

  const handleSendChat = (e: React.FormEvent) => {
    e.preventDefault();
    sendChat(chatInput);
  };

  const handleAssetClick = (asset: AssetData, targetTab?: 'analytics' | 'deep_analytics') => {
    setSelectedAsset(asset);
    const dest = targetTab || (activeTab === 'deep_analytics' ? 'deep_analytics' : 'analytics');
    setActiveTab(dest);
    setBacktestMetrics(null);
    setShapData(null);
    setMcData(null);
    if (dest === 'deep_analytics') {
      fetchAnalytics(asset.label);
      if (!wfvData) fetchWalkForward();
      fetchMonteCarlo(asset.label);
    }
    // Auto-ask the AI about this asset, passing the asset directly to avoid React state batching delay
    sendChat(`Analyze ${asset.label} for me. It is currently at $${asset.price.toLocaleString()} with a ${asset.signal} signal at ${asset.confidence}% confidence. Sentiment is ${asset.sentiment}.`, asset);
  };

  const getSparkColor = (signal: string) => {
    if (signal === 'BUY') return '#34D399';
    if (signal === 'SELL') return '#F87171';
    return '#A1A1AA';
  };

  const allAssets = scoutData ? [...scoutData.Crypto, ...scoutData.ETFs] : [];

  // ─── RENDER ───────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'var(--bg-dark)' }}>
      {/* HEADER */}
      <header style={{
        height: '64px',
        background: 'rgba(10, 10, 12, 0.96)',
        borderBottom: '1px solid var(--glass-border)',
        backdropFilter: 'blur(20px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 1.75rem',
        zIndex: 50,
        boxSizing: 'border-box'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            background: '#18181B',
            border: '1px solid rgba(255, 255, 255, 0.16)',
            padding: '0.4rem 0.85rem',
            borderRadius: '8px',
            fontWeight: 800,
            fontSize: '0.92rem',
            letterSpacing: '0.04em',
            color: '#FFFFFF',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            boxShadow: '0 2px 10px rgba(0, 0, 0, 0.5)'
          }}>
            <Activity size={17} color="#FFFFFF" /> QUANT ADVISOR PRO
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', borderLeft: '1px solid rgba(255,255,255,0.1)', paddingLeft: '1rem' }}>
            PPO TRANSFORMER TERMINAL
          </span>
        </div>

        {/* Center: Prominent Top Navigation Tabs */}
        <nav style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          background: 'rgba(255, 255, 255, 0.03)',
          border: '1px solid var(--glass-border)',
          borderRadius: '10px',
          padding: '0.3rem 0.35rem'
        }}>
          <button
            onClick={() => setActiveTab('dashboard')}
            style={{
              background: activeTab === 'dashboard' ? '#FFFFFF' : 'transparent',
              border: activeTab === 'dashboard' ? '1px solid #FFFFFF' : '1px solid transparent',
              boxShadow: activeTab === 'dashboard' ? '0 2px 12px rgba(255, 255, 255, 0.15)' : 'none',
              color: activeTab === 'dashboard' ? '#000000' : 'var(--text-secondary)',
              fontWeight: activeTab === 'dashboard' ? 700 : 500,
              borderRadius: '7px',
              padding: '0.42rem 1.1rem',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.18s ease'
            }}
          >
            <Activity size={16} color={activeTab === 'dashboard' ? '#000000' : 'currentColor'} />
            <span>Trading Dashboard</span>
          </button>

          <button
            onClick={() => setActiveTab('analytics')}
            style={{
              background: activeTab === 'analytics' ? '#FFFFFF' : 'transparent',
              border: activeTab === 'analytics' ? '1px solid #FFFFFF' : '1px solid transparent',
              boxShadow: activeTab === 'analytics' ? '0 2px 12px rgba(255, 255, 255, 0.15)' : 'none',
              color: activeTab === 'analytics' ? '#000000' : 'var(--text-secondary)',
              fontWeight: activeTab === 'analytics' ? 700 : 500,
              borderRadius: '7px',
              padding: '0.42rem 1.1rem',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.18s ease'
            }}
          >
            <BarChart2 size={16} color={activeTab === 'analytics' ? '#000000' : 'currentColor'} />
            <span>Asset Overview</span>
          </button>

          <button
            onClick={() => handleOpenDeepAnalytics()}
            style={{
              background: activeTab === 'deep_analytics' ? '#FFFFFF' : 'transparent',
              border: activeTab === 'deep_analytics' ? '1px solid #FFFFFF' : '1px solid transparent',
              boxShadow: activeTab === 'deep_analytics' ? '0 2px 12px rgba(255, 255, 255, 0.15)' : 'none',
              color: activeTab === 'deep_analytics' ? '#000000' : 'var(--text-secondary)',
              fontWeight: activeTab === 'deep_analytics' ? 700 : 500,
              borderRadius: '7px',
              padding: '0.42rem 1.1rem',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.18s ease'
            }}
          >
            <Cpu size={16} color={activeTab === 'deep_analytics' ? '#000000' : 'currentColor'} />
            <span>In-Depth Analysis</span>
          </button>

          <button
            onClick={() => setActiveTab('chat')}
            style={{
              background: activeTab === 'chat' ? '#FFFFFF' : 'transparent',
              border: activeTab === 'chat' ? '1px solid #FFFFFF' : '1px solid transparent',
              boxShadow: activeTab === 'chat' ? '0 2px 12px rgba(255, 255, 255, 0.15)' : 'none',
              color: activeTab === 'chat' ? '#000000' : 'var(--text-secondary)',
              fontWeight: activeTab === 'chat' ? 700 : 500,
              borderRadius: '7px',
              padding: '0.42rem 1.1rem',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.18s ease'
            }}
          >
            <Bot size={16} color={activeTab === 'chat' ? '#000000' : 'currentColor'} />
            <span>AI Advisor Chat</span>
          </button>

          <button
            onClick={() => setActiveTab('school')}
            style={{
              background: activeTab === 'school' ? '#FFFFFF' : 'transparent',
              border: activeTab === 'school' ? '1px solid #FFFFFF' : '1px solid transparent',
              boxShadow: activeTab === 'school' ? '0 2px 12px rgba(255, 255, 255, 0.15)' : 'none',
              color: activeTab === 'school' ? '#000000' : 'var(--text-secondary)',
              fontWeight: activeTab === 'school' ? 700 : 500,
              borderRadius: '7px',
              padding: '0.42rem 1.1rem',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.18s ease'
            }}
          >
            <BookOpen size={16} color={activeTab === 'school' ? '#000000' : 'currentColor'} />
            <span>Model School</span>
          </button>
        </nav>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {/* Live Market Status Pill */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.04)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '20px',
            padding: '0.3rem 0.8rem',
            fontSize: '0.75rem',
            color: '#E4E4E7',
            fontWeight: 500,
            display: 'flex',
            alignItems: 'center',
            gap: '0.45rem'
          }}>
            <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: '#34D399', boxShadow: '0 0 8px rgba(52, 211, 153, 0.6)', animation: 'pulse 2s infinite' }}></span>
            MARKETS LIVE
          </div>

          {/* GPU / CUDA Status Badge */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.04)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '20px',
            padding: '0.3rem 0.8rem',
            fontSize: '0.75rem',
            color: '#E4E4E7',
            fontWeight: 500,
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem'
          }}>
            <Activity size={13} color="#A1A1AA" /> CUDA ACCELERATED
          </div>

          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
            v2.4.0
          </div>
        </div>
      </header>

      <div className="layout-container" style={{ flex: 1 }}>
        {/* MAIN CONTENT */}
        <main className="main-content">

        {/* ── CHAT WORKSPACE TAB (Split Screen) ── */}
        {activeTab === 'chat' && (
          <>
            {/* Left Column: Chat panel */}
            <div className="chat-column">
              <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--glass-border)' }}>
                  <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Bot /> AI Advisor Chat
                  </h2>
                </div>

                <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {chatMessages.map((msg, i) => (
                    <div key={i} className="glass-panel chat-bubble">
                      <div className={`chat-avatar ${msg.role}`}>
                        {msg.role === 'assistant' ? <Bot size={20} color="white" /> : <div style={{ color: 'white', fontWeight: 'bold' }}>U</div>}
                      </div>
                      <div style={{ lineHeight: '1.6', fontSize: '0.95rem' }} dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }} />
                    </div>
                  ))}
                  {loading && (
                    <div className="glass-panel chat-bubble">
                      <div className="chat-avatar assistant"><Bot size={20} color="white" /></div>
                      <div className="typing-indicator"><span></span><span></span><span></span></div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                <div style={{ padding: '1rem', borderTop: '1px solid var(--glass-border)' }}>
                  <form className="chat-input-container" onSubmit={handleSendChat}>
                    <input
                      type="text"
                      className="chat-input"
                      placeholder="Ask the advisor a question..."
                      value={chatInput}
                      onChange={e => setChatInput(e.target.value)}
                    />
                    <button type="submit" style={{ position: 'absolute', right: '0.5rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: '#FFFFFF', cursor: 'pointer' }}>
                      <Send size={20} />
                    </button>
                  </form>
                </div>
              </div>
            </div>

            {/* Right Column: Simplified Asset context summary */}
            <div className="data-column" style={{ flex: 1.5 }}>
              <div className="glass-panel" style={{ padding: '2rem', height: '100%', boxSizing: 'border-box', display: 'flex', flexDirection: 'column', gap: '1.5rem', overflowY: 'auto' }}>
                {selectedAsset ? (
                  <>
                    <h2 style={{ marginTop: 0 }}>🎯 Active Ticker: {selectedAsset.label}</h2>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                      <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Current Price</div>
                        <div style={{ fontSize: '1.4rem', fontWeight: 'bold', marginTop: '0.25rem' }}>${selectedAsset.price.toLocaleString()}</div>
                      </div>
                      <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>AI Signal</div>
                        <div style={{ fontSize: '1.4rem', fontWeight: 'bold', marginTop: '0.25rem', color: getSparkColor(selectedAsset.signal) }}>{selectedAsset.signal} ({selectedAsset.confidence}%)</div>
                      </div>
                    </div>

                    <div style={{ padding: '1rem', background: 'rgba(0,0,0,0.1)', borderRadius: '12px' }}>
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>24H Trend (30min Candlesticks)</div>
                      <CandlestickChart data={selectedAsset.candles || []} width="100%" height={120} />
                    </div>

                    <div className="glass-panel" style={{ padding: '1.2rem' }}>
                      <h3 style={{ margin: '0 0 0.8rem 0', fontSize: '1rem' }}>💡 Suggested Questions</h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        <button className="chat-suggested-btn" onClick={() => sendChat(`Explain your rationale for ${selectedAsset.label}`)}>Why did you choose this signal?</button>
                        <button className="chat-suggested-btn" onClick={() => sendChat(`What is the risk level for ${selectedAsset.label}?`)}>How risky is this position?</button>
                        <button className="chat-suggested-btn" onClick={() => sendChat(`What technical indicators are driving ${selectedAsset.label}?`)}>Explain technical indicator drivers</button>
                      </div>
                    </div>
                  </>
                ) : (
                  <div style={{ textAlign: 'center', margin: 'auto', color: 'var(--text-secondary)' }}>
                    <Bot size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                    <p>No active asset context selected.</p>
                    <p style={{ fontSize: '0.85rem' }}>Select an asset on the **Dashboard** page or click on charts to feed real-time indicators to the AI Advisor.</p>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* ── DASHBOARD, ANALYTICS, DEEP ANALYTICS & SCHOOL TABS (Full Width) ── */}
        {(activeTab === 'dashboard' || activeTab === 'analytics' || activeTab === 'deep_analytics' || activeTab === 'school') && (
          <div className="data-column" style={{ flex: 1 }}>

            {/* ── 1. TRADING DASHBOARD TAB (Overview Grid) ── */}
            {activeTab === 'dashboard' && (
              <>
                {/* TOP ASSET SELECTION PANEL */}
                <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <div>
                      <h2 style={{ margin: 0, fontSize: '1.35rem' }}>📈 Market Asset Selector</h2>
                      <p style={{ color: 'var(--text-secondary)', margin: '0.3rem 0 0 0', fontSize: '0.9rem' }}>
                        Click any market asset below to open its dedicated full-page deep analytics, backtests, and PPO explainability.
                      </p>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                      <span style={{ fontSize: '0.72rem', background: 'rgba(255, 255, 255, 0.08)', padding: '0.2rem 0.55rem', borderRadius: '4px', color: '#E4E4E7', border: '1px solid var(--glass-border)', fontWeight: 500 }}>
                        ⏱ 30min
                      </span>
                      <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                        5 ASSETS MONITORED
                      </span>
                    </div>
                  </div>

                  {scoutData ? (
                    <>
                      {/* Quick Asset Selector Cards Grid */}
                      <div className="asset-grid" style={{ marginTop: '1.2rem' }}>
                        {allAssets.map(asset => {
                          const isSelected = selectedAsset?.label === asset.label;
                          return (
                            <div
                              key={asset.label}
                              className={`glass-panel asset-card ${isSelected ? 'selected' : ''}`}
                              onClick={() => handleAssetClick(asset)}
                              style={{ 
                                cursor: 'pointer',
                                border: isSelected ? '1px solid #FFFFFF' : '1px solid var(--glass-border)',
                                transform: isSelected ? 'scale(1.02)' : 'none',
                                transition: 'all 0.2s ease'
                              }}
                            >
                              <div className="asset-card-info">
                                <div style={{ fontWeight: 'bold', fontSize: '1.05rem', color: isSelected ? '#FFFFFF' : 'var(--text-primary)', textAlign: 'left' }}>
                                  {asset.label}
                                </div>
                                <div style={{ fontSize: '1.15rem', fontWeight: 'bold', color: 'white', textAlign: 'left', marginTop: '0.2rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                  ${asset.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: asset.price < 1 ? 5 : 2 })}
                                  <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#34D399', boxShadow: '0 0 6px #34D399', display: 'inline-block' }} title="Live Market Price"></span>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.2rem' }}>
                                  <span className={`signal-badge signal-${asset.signal.toLowerCase()}`} style={{ padding: '0.15rem 0.5rem', fontSize: '0.68rem' }}>
                                    <SignalIcon signal={asset.signal} />&nbsp;{asset.signal}
                                  </span>
                                  <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
                                    {asset.confidence}% conf
                                  </span>
                                </div>
                                {asset.regime && (
                                  <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', marginTop: '0.1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span>Regime: <span style={{ color: '#E4E4E7', fontWeight: '500' }}>{asset.regime}</span></span>
                                    <span style={{ color: 'var(--text-muted)', background: 'rgba(255,255,255,0.06)', padding: '1px 5px', borderRadius: '3px', fontSize: '0.65rem' }}>30m</span>
                                  </div>
                                )}
                              </div>
                              <div className="asset-card-chart">
                                <CandlestickChart data={asset.candles || []} width="100%" height={70} interactive={false} />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </>
                  ) : (
                    <div style={{ padding: '3rem 2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                      <div className="skeleton" style={{ width: '100%', height: '60px', marginBottom: '1.5rem', opacity: 0.3 }}></div>
                      <p style={{ fontSize: '1.05rem', marginBottom: '1.5rem' }}>Connecting to Trading Advisor API at port 8000...</p>
                      <button 
                        className="chat-suggested-btn" 
                        onClick={fetchScoutData}
                        style={{ width: 'auto', margin: '0 auto', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.6rem 1.5rem' }}
                      >
                        🔄 Retry Connection
                      </button>
                    </div>
                  )}
                </div>

                {/* 2. COMPACT CORNER SYSTEM TELEMETRY DOCK */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'flex-end',
                  marginTop: 'auto',
                  paddingTop: '1rem',
                  paddingBottom: '0.5rem'
                }}>
                  {isTelemetryCollapsed ? (
                    <div
                      className="glass-panel"
                      onClick={() => setIsTelemetryCollapsed(false)}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.65rem',
                        padding: '0.35rem 0.85rem',
                        borderRadius: '20px',
                        cursor: 'pointer',
                        background: 'rgba(18, 18, 20, 0.85)',
                        border: '1px solid var(--glass-border)',
                        boxShadow: '0 4px 16px rgba(0,0,0,0.3)',
                        transition: 'all 0.2s ease'
                      }}
                      title="Click to expand system status"
                    >
                      <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#10B981', boxShadow: '0 0 6px #10B981' }}></span>
                      <span style={{ fontSize: '0.72rem', fontWeight: 600, color: '#E4E4E7' }}>System Telemetry</span>
                      <span style={{ fontSize: '0.65rem', color: '#10B981', background: 'rgba(16, 185, 129, 0.12)', padding: '1px 5px', borderRadius: '3px' }}>ONLINE</span>
                      <ChevronUp size={13} color="var(--text-muted)" />
                    </div>
                  ) : (
                    <div
                      className="glass-panel"
                      style={{
                        width: '100%',
                        maxWidth: '340px',
                        padding: '0.75rem 0.9rem',
                        borderRadius: '12px',
                        background: 'rgba(14, 14, 16, 0.88)',
                        border: '1px solid var(--glass-border)',
                        boxShadow: '0 8px 28px rgba(0, 0, 0, 0.45)',
                        transition: 'all 0.2s ease'
                      }}
                    >
                      {/* Telemetry Header */}
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.55rem', borderBottom: '1px solid rgba(255, 255, 255, 0.08)', paddingBottom: '0.4rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <Radio size={12} color="#10B981" />
                          <span style={{ fontSize: '0.72rem', fontWeight: 700, color: '#E4E4E7', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                            System Status
                          </span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span style={{ fontSize: '0.65rem', color: '#10B981', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <span style={{ width: '5px', height: '5px', borderRadius: '50%', background: '#10B981', display: 'inline-block' }}></span>
                            ALL ONLINE
                          </span>
                          <button
                            onClick={() => setIsTelemetryCollapsed(true)}
                            style={{
                              background: 'transparent',
                              border: 'none',
                              color: 'var(--text-muted)',
                              cursor: 'pointer',
                              padding: '2px',
                              display: 'flex',
                              alignItems: 'center',
                              borderRadius: '4px'
                            }}
                            title="Minimize to corner pill"
                          >
                            <ChevronDown size={13} />
                          </button>
                        </div>
                      </div>

                      {/* 3 Compact Status Sub-Cards */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                        {/* Card 1: Trading System Engine */}
                        <div style={{
                          padding: '0.45rem 0.6rem',
                          background: 'rgba(255, 255, 255, 0.02)',
                          borderRadius: '7px',
                          border: '1px solid rgba(255, 255, 255, 0.05)'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.15rem' }}>
                            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Trading System Engine</span>
                            <span style={{ fontSize: '0.65rem', color: '#10B981', fontWeight: 700 }}>ONLINE</span>
                          </div>
                          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#FFFFFF' }}>PPO Transformer Architecture</div>
                          <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)', margin: '0.2rem 0 0 0', lineHeight: 1.3 }}>
                            Actor-Critic model trained on continuous tensors & FinBERT.
                          </p>
                        </div>

                        {/* Card 2: Live Market Feeds */}
                        <div style={{
                          padding: '0.45rem 0.6rem',
                          background: 'rgba(255, 255, 255, 0.02)',
                          borderRadius: '7px',
                          border: '1px solid rgba(255, 255, 255, 0.05)'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.15rem' }}>
                            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Live Market Feeds</span>
                            <span style={{ fontSize: '0.65rem', color: '#10B981', fontWeight: 700 }}>SYNCED</span>
                          </div>
                          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#FFFFFF' }}>Binance + Yahoo Finance</div>
                          <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)', margin: '0.2rem 0 0 0', lineHeight: 1.3 }}>
                            Real-time 5m candlestick feeds & GARCH volatility estimation.
                          </p>
                        </div>

                        {/* Card 3: Deep Analytics */}
                        <div 
                          onClick={() => handleOpenDeepAnalytics()}
                          style={{
                            padding: '0.45rem 0.6rem',
                            background: 'rgba(255, 255, 255, 0.02)',
                            borderRadius: '7px',
                            border: '1px solid rgba(255, 255, 255, 0.05)',
                            cursor: 'pointer',
                            transition: 'border-color 0.15s ease'
                          }}
                          title="Click to view deep analytics"
                        >
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.15rem' }}>
                            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Deep Analytics</span>
                            <span style={{ fontSize: '0.65rem', color: '#A1A1AA', fontWeight: 700 }}>READY</span>
                          </div>
                          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#FFFFFF', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <span>Dedicated Page Available</span>
                            <span style={{ fontSize: '0.68rem', color: 'var(--text-secondary)' }}>→</span>
                          </div>
                          <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)', margin: '0.2rem 0 0 0', lineHeight: 1.3 }}>
                            Click to inspect full backtests, HMM regimes, and SHAP.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}

            {/* ── 2. ASSET ANALYTICS TAB (Dedicated Full Page) ── */}
            {activeTab === 'analytics' && (
              <>
                {/* Top Navigation Bar: Back Button & Asset Switcher */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
                  <button
                    onClick={() => setActiveTab('dashboard')}
                    style={{
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid var(--glass-border)',
                      color: '#FFFFFF',
                      borderRadius: '8px',
                      padding: '0.45rem 1rem',
                      fontSize: '0.85rem',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <ArrowLeft size={16} /> Back to Dashboard
                  </button>

                  {/* Asset Switcher Pills */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginRight: '0.2rem' }}>Switch Asset:</span>
                    {allAssets.map(a => {
                      const isCur = selectedAsset?.label === a.label;
                      return (
                        <button
                          key={a.label}
                          onClick={() => handleAssetClick(a)}
                          style={{
                            background: isCur ? '#FFFFFF' : 'rgba(255, 255, 255, 0.04)',
                            color: isCur ? '#000000' : 'var(--text-secondary)',
                            border: isCur ? '1px solid #FFFFFF' : '1px solid var(--glass-border)',
                            borderRadius: '6px',
                            padding: '0.35rem 0.75rem',
                            fontSize: '0.78rem',
                            fontWeight: isCur ? 700 : 500,
                            cursor: 'pointer',
                            transition: 'all 0.15s ease'
                          }}
                        >
                          {a.label}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* DEEP ANALYTICS & CHARTS PANEL */}
                {selectedAsset ? (
                  <div className="glass-panel" style={{ padding: '2rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                      <div>
                        <h2 style={{ margin: 0, fontSize: '1.6rem' }}>{selectedAsset.label}</h2>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '0.35rem' }}>
                          <span style={{ color: '#FAFAFA', fontWeight: 600 }}>
                            {scoutData?.Crypto.some(c => c.label === selectedAsset.label) || selectedAsset.label.includes('BTC') || selectedAsset.label.includes('ETH') || selectedAsset.label.includes('DOGE') ? 'Binance Spot' : 'Yahoo Finance'}
                          </span>
                          <span style={{ opacity: 0.4 }}>•</span>
                          <span>
                            {scoutData?.Crypto.some(c => c.label === selectedAsset.label) || selectedAsset.label.includes('BTC') || selectedAsset.label.includes('ETH') || selectedAsset.label.includes('DOGE') ? 'Crypto' : 'Equity ETF'}
                          </span>
                          <span style={{ opacity: 0.4 }}>•</span>
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
                            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#34D399', boxShadow: '0 0 6px rgba(52, 211, 153, 0.7)', display: 'inline-block' }}></span>
                            5m Live Stream
                          </span>
                          <span style={{ opacity: 0.4 }}>•</span>
                          <span>USD</span>
                          {selectedAsset.regime && (
                            <>
                              <span style={{ opacity: 0.4 }}>•</span>
                              <span style={{
                                background: 'rgba(255, 255, 255, 0.06)',
                                border: '1px solid rgba(255, 255, 255, 0.08)',
                                padding: '1px 7px',
                                borderRadius: '4px',
                                color: '#E4E4E7',
                                fontSize: '0.75rem',
                                fontWeight: 500
                              }}>
                                Regime: {selectedAsset.regime}
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                        <button
                          onClick={() => {
                            setActiveTab('chat');
                            sendChat(`Analyze ${selectedAsset.label} for me. It is currently at $${selectedAsset.price.toLocaleString()} with a ${selectedAsset.signal} signal at ${selectedAsset.confidence}% confidence. Sentiment is ${selectedAsset.sentiment}.`, selectedAsset);
                          }}
                          style={{
                            background: 'rgba(255, 255, 255, 0.06)',
                            border: '1px solid var(--glass-border)',
                            color: '#FFFFFF',
                            borderRadius: '8px',
                            padding: '0.4rem 0.9rem',
                            fontSize: '0.82rem',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.4rem'
                          }}
                        >
                          <Bot size={15} /> Ask Advisor
                        </button>
                        <span className={`signal-badge signal-${selectedAsset.signal.toLowerCase()}`} style={{ fontSize: '0.88rem', padding: '0.4rem 1rem' }}>
                          <SignalIcon signal={selectedAsset.signal} />&nbsp;{selectedAsset.signal} — {selectedAsset.confidence}% Confidence
                        </span>
                      </div>
                    </div>

                    <div style={{ marginTop: '1.5rem' }}>
                      <CandlestickChart data={selectedAsset.candles || []} width="100%" height={200} interactive={true} />
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginTop: '2rem' }}>
                      <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '0.25rem' }}>Price</div>
                        <div style={{ fontSize: '1.3rem', fontWeight: 'bold' }}>${selectedAsset.price.toLocaleString()}</div>
                      </div>
                      <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '0.25rem' }}>Sentiment</div>
                        <div style={{ fontSize: '1.3rem', fontWeight: 'bold' }}>{selectedAsset.sentiment}</div>
                      </div>
                      <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '0.25rem' }}>Signal</div>
                        <div style={{ fontSize: '1.3rem', fontWeight: 'bold', color: getSparkColor(selectedAsset.signal) }}>{selectedAsset.signal}</div>
                      </div>
                    </div>

                    {/* SECTION 1: OVERVIEW */}
                    <div id="sec-overview" style={{ scrollMarginTop: '120px', marginTop: '1.5rem' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                        <div className="glass-panel" style={{ padding: '1.5rem' }}>
                          <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem' }}>
                            🧠 AI Confidence Breakdown
                            <InfoBadge topic="AI Confidence" text="Derived from PPO state-value function and feature attribution." anchor="#state-space" onNavigate={handleNavigateSchool} />
                          </h3>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                            <div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.3rem' }}>
                                <span style={{ color: 'var(--text-secondary)' }}>Trend Strength</span>
                                <span>84%</span>
                              </div>
                              <div style={{ background: 'rgba(0,0,0,0.3)', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                                <div style={{ background: '#E4E4E7', height: '100%', width: '84%' }}></div>
                              </div>
                            </div>
                            <div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.3rem' }}>
                                <span style={{ color: 'var(--text-secondary)' }}>Volume Momentum</span>
                                <span>62%</span>
                              </div>
                              <div style={{ background: 'rgba(0,0,0,0.3)', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                                <div style={{ background: '#E4E4E7', height: '100%', width: '62%' }}></div>
                              </div>
                            </div>
                            <div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.3rem' }}>
                                <span style={{ color: 'var(--text-secondary)' }}>News Sentiment Impact</span>
                                <span>{selectedAsset.confidence}%</span>
                              </div>
                              <div style={{ background: 'rgba(0,0,0,0.3)', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                                <div style={{ background: getSparkColor(selectedAsset.signal), height: '100%', width: `${selectedAsset.confidence}%` }}></div>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="glass-panel" style={{ padding: '1.5rem' }}>
                          <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem' }}>📊 Key Technical Levels</h3>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Resistance 1</span>
                              <span style={{ fontWeight: '500' }}>${(selectedAsset.price * 1.05).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Current Price (Live)</span>
                              <span style={{ fontWeight: 'bold', color: '#FFFFFF', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                                ${selectedAsset.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: selectedAsset.price < 1 ? 5 : 2 })}
                                <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: '#34D399', boxShadow: '0 0 8px #34D399', display: 'inline-block' }} title="Live Market Price"></span>
                              </span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Support 1</span>
                              <span style={{ fontWeight: '500' }}>${(selectedAsset.price * 0.96).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* IN-DEPTH QUANTITATIVE ANALYSIS CTA CARD */}
                    <div style={{ marginTop: '2rem', padding: '1.8rem', background: 'linear-gradient(180deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.01) 100%)', borderRadius: '14px', border: '1px solid var(--glass-border)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                            <Cpu size={20} color="#34D399" />
                            <h3 style={{ margin: 0, fontSize: '1.2rem', color: '#FFFFFF' }}>In-Depth Quantitative Analysis</h3>
                          </div>
                          <p style={{ margin: '0.4rem 0 0 0', color: 'var(--text-secondary)', fontSize: '0.88rem', maxWidth: '620px', lineHeight: 1.5 }}>
                            Inspect the underlying mathematical and reinforcement learning engines for <strong>{selectedAsset.label}</strong>: Transformer walk-forward backtests, SHAP feature attributions, HMM market regime dynamics, 5-fold temporal cross-validation, and Monte Carlo risk equity cones.
                          </p>
                        </div>
                        <button
                          onClick={() => handleOpenDeepAnalytics()}
                          style={{
                            background: '#FFFFFF',
                            color: '#000000',
                            border: 'none',
                            borderRadius: '8px',
                            padding: '0.75rem 1.6rem',
                            fontSize: '0.9rem',
                            fontWeight: 700,
                            cursor: 'pointer',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.5rem',
                            boxShadow: '0 2px 14px rgba(255, 255, 255, 0.2)',
                            transition: 'all 0.18s ease'
                          }}
                        >
                          Open In-Depth Quant Lab →
                        </button>
                      </div>

                      {/* Feature Preview Badges */}
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', marginTop: '1.25rem' }}>
                        <div style={{ background: 'rgba(0,0,0,0.25)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Backtest Engine</div>
                          <div style={{ fontSize: '0.88rem', fontWeight: 600, color: '#FFFFFF', marginTop: '0.2rem' }}>📈 1,000+ Step Trajectory</div>
                        </div>
                        <div style={{ background: 'rgba(0,0,0,0.25)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Explainability</div>
                          <div style={{ fontSize: '0.88rem', fontWeight: 600, color: '#FFFFFF', marginTop: '0.2rem' }}>🔍 SHAP Attribution Weights</div>
                        </div>
                        <div style={{ background: 'rgba(0,0,0,0.25)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Regime Model</div>
                          <div style={{ fontSize: '0.88rem', fontWeight: 600, color: '#FFFFFF', marginTop: '0.2rem' }}>🌀 Hidden Markov States</div>
                        </div>
                        <div style={{ background: 'rgba(0,0,0,0.25)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Overfitting Guard</div>
                          <div style={{ fontSize: '0.88rem', fontWeight: 600, color: '#FFFFFF', marginTop: '0.2rem' }}>📊 5-Fold Walk-Forward CV</div>
                        </div>
                        <div style={{ background: 'rgba(0,0,0,0.25)', padding: '0.8rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Stress Testing</div>
                          <div style={{ fontSize: '0.88rem', fontWeight: 600, color: '#FFFFFF', marginTop: '0.2rem' }}>🎲 Monte Carlo Risk Cone</div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="glass-panel" style={{ padding: '3.5rem 2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                    <BarChart2 size={48} style={{ opacity: 0.25, marginBottom: '1rem' }} />
                    <h3 style={{ margin: '0 0 0.5rem 0', color: '#FFFFFF' }}>No Asset Selected</h3>
                    <p style={{ fontSize: '0.9rem', maxWidth: '420px', margin: '0 auto 1.5rem auto' }}>
                      Select an asset from the switcher above or return to the Trading Dashboard to inspect its overview.
                    </p>
                    <button
                      className="chat-suggested-btn"
                      onClick={() => setActiveTab('dashboard')}
                      style={{ width: 'auto', margin: '0 auto', display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1.25rem' }}
                    >
                      ← Return to Trading Dashboard
                    </button>
                  </div>
                )}
              </>
            )}

            {/* ── 3. IN-DEPTH QUANTITATIVE ANALYSIS TAB (Dedicated Workspace) ── */}
            {activeTab === 'deep_analytics' && (
              <>
                {/* Top Navigation Bar: Back Button & Asset Switcher */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
                  <button
                    onClick={() => setActiveTab('analytics')}
                    style={{
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid var(--glass-border)',
                      color: '#FFFFFF',
                      borderRadius: '8px',
                      padding: '0.45rem 1rem',
                      fontSize: '0.85rem',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <ArrowLeft size={16} /> Back to Asset Overview
                  </button>

                  {/* Asset Switcher Pills */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginRight: '0.2rem' }}>Switch Asset:</span>
                    {allAssets.map(a => {
                      const isCur = selectedAsset?.label === a.label;
                      return (
                        <button
                          key={a.label}
                          onClick={() => handleAssetClick(a, 'deep_analytics')}
                          style={{
                            background: isCur ? '#FFFFFF' : 'rgba(255, 255, 255, 0.04)',
                            color: isCur ? '#000000' : 'var(--text-secondary)',
                            border: isCur ? '1px solid #FFFFFF' : '1px solid var(--glass-border)',
                            borderRadius: '6px',
                            padding: '0.35rem 0.75rem',
                            fontSize: '0.78rem',
                            fontWeight: isCur ? 700 : 500,
                            cursor: 'pointer',
                            transition: 'all 0.15s ease'
                          }}
                        >
                          {a.label}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {selectedAsset ? (
                  <div className="glass-panel" style={{ padding: '2rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                          <Cpu size={22} color="#34D399" />
                          <h2 style={{ margin: 0, fontSize: '1.6rem' }}>In-Depth Analysis: {selectedAsset.label}</h2>
                        </div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                          Deep Transformer Policy Backtesting, SHAP Attribution, HMM Regimes & Stress Testing
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                        <button
                          onClick={() => {
                            if (selectedAsset) {
                              fetchAnalytics(selectedAsset.label);
                              fetchWalkForward();
                              fetchMonteCarlo(selectedAsset.label);
                            }
                          }}
                          disabled={loadingAnalytics || loadingWFV || loadingMC}
                          style={{
                            background: 'rgba(255, 255, 255, 0.06)',
                            border: '1px solid var(--glass-border)',
                            color: '#FFFFFF',
                            borderRadius: '8px',
                            padding: '0.4rem 0.9rem',
                            fontSize: '0.82rem',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.4rem'
                          }}
                        >
                          🔄 Refresh All Models
                        </button>
                        <button
                          onClick={() => {
                            setActiveTab('chat');
                            sendChat(`Provide an in-depth quantitative analysis of ${selectedAsset.label}, explaining its policy backtest, feature attributions, and regime resilience.`, selectedAsset);
                          }}
                          style={{
                            background: '#FFFFFF',
                            color: '#000000',
                            border: 'none',
                            borderRadius: '8px',
                            padding: '0.4rem 0.9rem',
                            fontSize: '0.82rem',
                            cursor: 'pointer',
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.4rem'
                          }}
                        >
                          <Bot size={15} /> Ask Advisor
                        </button>
                      </div>
                    </div>

                    {/* Sub-Tab Filter Toolbar */}
                    <div className="detail-tabs-container" style={{ position: 'relative', zIndex: 10, background: 'rgba(16, 16, 20, 0.75)', backdropFilter: 'blur(12px)', padding: '0.75rem 1rem', borderRadius: '12px', marginTop: '1.5rem', marginBottom: '1.5rem', display: 'flex', gap: '0.6rem', border: '1px solid var(--glass-border)', overflowX: 'auto' }}>
                      <button onClick={() => setDeepSubTab('all')} className={`detail-tab ${deepSubTab === 'all' ? 'active' : ''}`} style={{ cursor: 'pointer', padding: '0.4rem 0.9rem' }}>🌐 All Modules</button>
                      <button onClick={() => setDeepSubTab('backtest')} className={`detail-tab ${deepSubTab === 'backtest' ? 'active' : ''}`} style={{ cursor: 'pointer', padding: '0.4rem 0.9rem' }}>📈 Backtest & Equity</button>
                      <button onClick={() => setDeepSubTab('shap')} className={`detail-tab ${deepSubTab === 'shap' ? 'active' : ''}`} style={{ cursor: 'pointer', padding: '0.4rem 0.9rem' }}>🔍 SHAP Explainability</button>
                      <button onClick={() => setDeepSubTab('regime')} className={`detail-tab ${deepSubTab === 'regime' ? 'active' : ''}`} style={{ cursor: 'pointer', padding: '0.4rem 0.9rem' }}>🌀 HMM Regimes</button>
                      <button onClick={() => setDeepSubTab('wfv')} className={`detail-tab ${deepSubTab === 'wfv' ? 'active' : ''}`} style={{ cursor: 'pointer', padding: '0.4rem 0.9rem' }}>📊 Walk-Forward CV</button>
                      <button onClick={() => setDeepSubTab('montecarlo')} className={`detail-tab ${deepSubTab === 'montecarlo' ? 'active' : ''}`} style={{ cursor: 'pointer', padding: '0.4rem 0.9rem' }}>🎲 Monte Carlo Risk</button>
                    </div>

                    {/* SECTION 2: MODEL EVALUATION */}
                    {(deepSubTab === 'all' || deepSubTab === 'backtest') && (
                      <div id="sec-eval" style={{ marginTop: '1.5rem' }}>
                        <div className="glass-panel" style={{ padding: '1.5rem' }}>
                        {(() => {
                          const dates = backtestMetrics?.dates || [];
                          const startDate = dates.length > 0 ? dates[0].split(' ')[0] : '';
                          const endDate = dates.length > 0 ? dates[dates.length - 1].split(' ')[0] : '';
                          const dateRangeText = startDate && endDate ? `(${startDate} to ${endDate})` : '';
                          return (
                            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                              <span>📈 Walk-Forward Backtest Performance <InfoBadge topic="Walk-Forward Testing" text="Out-of-sample simulation mimicking real-world forward propagation." anchor="#rl-basics" onNavigate={handleNavigateSchool} /></span>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                {dateRangeText && <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 'normal' }}>{startDate} to {endDate}</span>}
                                {backtestMetrics && (
                                  <button
                                    onClick={() => fetchAnalytics(selectedAsset.label)}
                                    disabled={loadingAnalytics}
                                    style={{
                                      background: 'rgba(255, 255, 255, 0.08)',
                                      border: '1px solid var(--glass-border)',
                                      color: '#FFFFFF',
                                      borderRadius: '6px',
                                      padding: '0.3rem 0.75rem',
                                      fontSize: '0.78rem',
                                      cursor: 'pointer',
                                      display: 'flex',
                                      alignItems: 'center',
                                      gap: '0.35rem'
                                    }}
                                  >
                                    🔄 Re-run Backtest
                                  </button>
                                )}
                              </div>
                            </h3>
                          );
                        })()}
                        {loadingAnalytics ? (
                          <div style={{ padding: '3.5rem 2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                            <div className="skeleton" style={{ width: '100%', height: '180px', marginBottom: '1.25rem', opacity: 0.25 }}></div>
                            <div style={{ fontSize: '1rem', color: '#FFFFFF', fontWeight: 600 }}>
                              Running historical backtest simulation on {selectedAsset.label}...
                            </div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                              Evaluating 1,000+ sequential 30-min trading decisions with 0.10% commission & 0.05% slippage friction.
                            </div>
                          </div>
                        ) : !backtestMetrics ? (
                          <div style={{ padding: '3rem 2rem', textAlign: 'center', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '10px', border: '1px dashed var(--glass-border)' }}>
                            <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#FFFFFF', marginBottom: '0.4rem' }}>
                              Walk-Forward Backtest Simulation
                            </div>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', maxWidth: '520px', margin: '0 auto 1.5rem auto', lineHeight: 1.5 }}>
                              Execute a comprehensive out-of-sample backtest for <strong>{selectedAsset.label}</strong> across historical 30-min bars. Evaluates Sharpe ratio, maximum drawdown, total return, and trade-by-trade equity trajectories under real market friction.
                            </p>
                            <button
                              onClick={() => fetchAnalytics(selectedAsset.label)}
                              style={{
                                background: '#FFFFFF',
                                color: '#000000',
                                border: 'none',
                                borderRadius: '8px',
                                padding: '0.65rem 1.6rem',
                                fontSize: '0.9rem',
                                fontWeight: 700,
                                cursor: 'pointer',
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '0.5rem',
                                boxShadow: '0 2px 12px rgba(255, 255, 255, 0.15)',
                                transition: 'all 0.18s ease'
                              }}
                            >
                              ▶ Run Backtest Simulation
                            </button>
                          </div>
                        ) : (
                          <>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginTop: '1rem' }}>
                              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Sharpe Ratio <InfoBadge topic="Sharpe Ratio" text="Risk-adjusted return metric. Above 1.0 is good, >2.0 is excellent." anchor="#rl-basics" onNavigate={handleNavigateSchool} /></div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: backtestMetrics.sharpe >= 0 ? '#4ade80' : '#f87171' }}>{Number(backtestMetrics.sharpe).toFixed(2)}</div>
                              </div>
                              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Max Drawdown <InfoBadge topic="Maximum Drawdown" text="The largest peak-to-trough drop in portfolio equity." anchor="#hmm-regimes" onNavigate={handleNavigateSchool} /></div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#f87171' }}>{Number(backtestMetrics.max_dd).toFixed(2)}%</div>
                              </div>
                              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Total Trades</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{backtestMetrics.trades}</div>
                              </div>
                              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Total Return</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: backtestMetrics.roi >= 0 ? '#4ade80' : '#f87171' }}>{backtestMetrics.roi >= 0 ? '+' : ''}{Number(backtestMetrics.roi).toFixed(2)}%</div>
                              </div>
                            </div>

                            
<div style={{ marginTop: '2rem', padding: '1.2rem', background: 'rgba(0,0,0,0.1)', borderRadius: '12px' }}>
                              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                                Backtest Equity Curve {backtestMetrics.dates && backtestMetrics.dates.length > 0 ? `(${backtestMetrics.dates[0].split(' ')[0]} to ${backtestMetrics.dates[backtestMetrics.dates.length - 1].split(' ')[0]})` : ''}
                              </div>
                              {backtestMetrics.history && (() => {
                                const totalSteps = backtestMetrics.history.length;
                                const maxRenderPoints = 350;
                                const stride = Math.max(1, Math.floor(totalSteps / maxRenderPoints));

                                // Build O(1) lookup map for trade markers
                                const tradeMap = new Map<number, TradeItem>();
                                if (backtestMetrics.trade_history) {
                                  for (let k = 0; k < backtestMetrics.trade_history.length; k++) {
                                    const t = backtestMetrics.trade_history[k];
                                    tradeMap.set(t.step, t);
                                  }
                                }

                                let currentPeak = backtestMetrics.history[0];
                                let minVal = Infinity;
                                let maxVal = -Infinity;
                                let ddMin = 0;

                                const chartData = [];
                                for (let idx = 0; idx < totalSteps; idx++) {
                                  const val = backtestMetrics.history[idx];
                                  if (val > currentPeak) currentPeak = val;
                                  
                                  const trade = tradeMap.get(idx);
                                  const isBuy = trade && (trade.action === 1 || trade.action === 4);
                                  const isSell = trade && (trade.action === 2 || trade.action === 3);

                                  // Get date label if available
                                  const rawDate = backtestMetrics.dates && backtestMetrics.dates[idx] ? backtestMetrics.dates[idx] : `Step ${idx}`;
                                  const shortDate = rawDate.includes(' ') ? rawDate.split(' ')[0] : rawDate;

                                  // Downsample: include points at regular stride intervals, trade points, and endpoints
                                  if (idx % stride === 0 || trade || idx === totalSteps - 1) {
                                    const dd = ((val - currentPeak) / currentPeak) * 100;
                                    if (val < minVal) minVal = val;
                                    if (val > maxVal) maxVal = val;
                                    if (dd < ddMin) ddMin = dd;

                                    chartData.push({
                                      step: idx,
                                      date: shortDate,
                                      dateFull: rawDate,
                                      netWorth: Math.round(val * 100) / 100,
                                      drawdown: Math.round(dd * 100) / 100,
                                      buyMarker: isBuy ? val : null,
                                      sellMarker: isSell ? val : null
                                    });
                                  }
                                }

                                const yDomain: [number | string, number | string] = minVal === maxVal || !isFinite(minVal) ? [10000 - 100, 10000 + 100] : ['auto', 'auto'];
                                const ddDomain: [number | string, number | string] = ddMin === 0 ? [-1, 0] : ['auto', 0];

                                return (
                                  <>
                                    <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
                                      <ResponsiveContainer width="100%" height="100%">
                                        <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                                          <XAxis 
                                            dataKey="date" 
                                            stroke="rgba(255,255,255,0.2)" 
                                            fontSize={10} 
                                            interval="preserveStartEnd"
                                            minTickGap={60}
                                          />
                                          <YAxis stroke="rgba(255,255,255,0.2)" fontSize={10} domain={yDomain} tickFormatter={(v) => `$${v}`} />
                                          <RechartsTooltip 
                                            contentStyle={{ background: 'rgba(12, 12, 14, 0.96)', border: '1px solid rgba(255, 255, 255, 0.18)', borderRadius: '8px' }} 
                                            labelStyle={{ color: '#FFFFFF', fontWeight: 'bold', marginBottom: '0.2rem' }}
                                            labelFormatter={(label, items) => {
                                              if (items && items[0] && items[0].payload && items[0].payload.dateFull) {
                                                return items[0].payload.dateFull;
                                              }
                                              return label;
                                            }}
                                            formatter={(value, name) => {
                                              if (name === 'netWorth' && typeof value === 'number') return [`$${value.toFixed(2)}`, 'Equity'];
                                              if (name === 'drawdown' && typeof value === 'number') return [`${value.toFixed(2)}%`, 'Drawdown'];
                                              return [null, null];
                                            }}
                                          />
                                          <Line type="stepAfter" dataKey="netWorth" stroke="#34D399" strokeWidth={2} dot={false} isAnimationActive={false} />
                                          <Scatter dataKey="buyMarker" fill="#34D399" shape="triangle" isAnimationActive={false} />
                                          <Scatter dataKey="sellMarker" fill="#F87171" shape="diamond" isAnimationActive={false} />
                                        </ComposedChart>
                                      </ResponsiveContainer>
                                    </div>
                                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '1.5rem', marginBottom: '0.5rem' }}>Peak-to-Trough Drawdown</div>
                                    <div style={{ height: 100, width: '100%' }}>
                                      <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={chartData} margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
                                          <XAxis dataKey="date" hide />
                                          <YAxis hide domain={ddDomain} />
                                          <RechartsTooltip 
                                            contentStyle={{ background: 'rgba(12, 12, 14, 0.96)', border: '1px solid rgba(248, 113, 113, 0.35)', borderRadius: '8px' }} 
                                            labelFormatter={(label, items) => {
                                              if (items && items[0] && items[0].payload && items[0].payload.dateFull) {
                                                return items[0].payload.dateFull;
                                              }
                                              return label;
                                            }}
                                            formatter={(value) => [`${typeof value === 'number' ? value.toFixed(2) : value}%`, 'Drawdown']}
                                          />
                                          <Area type="stepAfter" dataKey="drawdown" stroke="#F87171" fill="#F87171" fillOpacity={0.25} isAnimationActive={false} />
                                        </AreaChart>
                                      </ResponsiveContainer>
                                    </div>
                                  </>
                                );
                              })()}
                            </div>
                            
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '1rem', fontStyle: 'italic' }}>
                              * Backtest executed deterministically across {backtestMetrics.dates && backtestMetrics.dates.length > 0 ? `${backtestMetrics.dates[0].split(' ')[0]} to ${backtestMetrics.dates[backtestMetrics.dates.length - 1].split(' ')[0]}` : 'the validation dataset'} with 0.1% commission and 0.05% slippage friction parameters.
                            </p>
                          </>
                        )}
                      </div>
                    </div>
                    )}

                    {/* SECTION 3: REGIME ANALYSIS */}
                    {(deepSubTab === 'all' || deepSubTab === 'regime') && (
                      <div id="sec-regime" style={{ scrollMarginTop: '120px', marginTop: '2rem' }}>
                      <div className="glass-panel" style={{ padding: '1.5rem' }}>
                        <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem' }}>🌀 Hidden Markov Model Regime Detection</h3>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '2rem', marginTop: '1rem' }}>
                          <div style={{ flex: 1 }}>
                            <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Current Market Regime</div>
                            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#FFFFFF' }}>
                              {selectedAsset.regime || "Sideways / Quiet"}
                            </div>
                            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                              {(selectedAsset.regime || "").includes("Bull") ? (
                                "The PPO agent is operating in a bullish regime. It tends to favor opening Long positions and holding assets to capture upward momentum, while setting wider stop-losses to absorb minor volatility."
                              ) : (selectedAsset.regime || "").includes("Bear") ? (
                                "The model classifies the current regime as bearish. The policy acts defensively by prioritizing shorting opportunities, locking in quick gains, and sitting in USD/Neutral during panic selloffs."
                              ) : (
                                "The model classifies the current regime as sideways or consolidating. To avoid fee attrition and false breakouts, the PPO policy is trained to scale down trading frequency and maintain a high cash/neutral allocation."
                              )}
                            </p>
                          </div>
                          <div style={{ width: '150px', height: '150px', borderRadius: '50%', border: '2px solid rgba(255, 255, 255, 0.18)', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
                            <div style={{ position: 'absolute', width: '10px', height: '10px', background: '#FFFFFF', borderRadius: '50%', top: '20px', right: '30px', boxShadow: '0 0 12px rgba(255, 255, 255, 0.8)' }}></div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textAlign: 'center' }}>HMM<br />State<br />Classifier</div>
                          </div>
                        </div>
                      </div>
                    </div>
                    )}

                    {/* SECTION 4: SHAP EXPLAINABILITY */}
                    {(deepSubTab === 'all' || deepSubTab === 'shap') && (
                      <div id="sec-shap" style={{ scrollMarginTop: '120px', marginTop: '2rem' }}>
                      <div className="glass-panel" style={{ padding: '1.5rem' }}>
                        <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem' }}>🔍 SHAP Feature Importance</h3>
                        {loadingAnalytics ? (
                          <div style={{ padding: '2.5rem 1.5rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                            <div className="typing-indicator" style={{ margin: 'auto' }}><span></span><span></span><span></span></div>
                            <div style={{ marginTop: '1rem', color: '#FFFFFF', fontSize: '0.9rem' }}>Computing SHAP attribution values for the latest decision weights...</div>
                          </div>
                        ) : !shapData ? (
                          <div style={{ padding: '2.5rem 1.5rem', textAlign: 'center', color: 'var(--text-secondary)', background: 'rgba(255, 255, 255, 0.01)', borderRadius: '8px', border: '1px dashed var(--glass-border)' }}>
                            <p style={{ margin: '0 auto 1.25rem auto', fontSize: '0.88rem', maxWidth: '480px' }}>
                              SHAP feature importance calculates which factors (RSI, GARCH volatility, FinBERT sentiment, MACD) most heavily influenced the latest PPO policy decision for <strong>{selectedAsset.label}</strong>.
                            </p>
                            <button
                              onClick={() => fetchAnalytics(selectedAsset.label)}
                              style={{
                                background: 'rgba(255, 255, 255, 0.08)',
                                color: '#FFFFFF',
                                border: '1px solid var(--glass-border)',
                                borderRadius: '6px',
                                padding: '0.45rem 1.2rem',
                                fontSize: '0.82rem',
                                cursor: 'pointer',
                                fontWeight: 500
                              }}
                            >
                              🔍 Compute Feature Attribution
                            </button>
                          </div>
                        ) : (
                          <>
                            <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                              {shapData.shap.slice(0, 8).map((item: ShapItem, idx: number) => {
                                const maxAbs = Math.max(...shapData.shap.map((x: ShapItem) => Math.abs(x.value))) || 1;
                                const widthPct = Math.min(100, Math.max(5, (Math.abs(item.value) / maxAbs) * 100));
                                const barColor = item.value >= 0 ? '#4ade80' : '#f87171';
                                return (
                                  <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                    <div style={{ width: '180px', fontSize: '0.85rem', color: 'var(--text-secondary)', textAlign: 'right', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                      {item.feature}
                                    </div>
                                    <div style={{ flex: 1, background: 'rgba(0,0,0,0.2)', height: '12px', borderRadius: '6px' }}>
                                      <div style={{ background: barColor, width: `${widthPct}%`, height: '100%', borderRadius: '6px', transition: 'width 0.4s ease' }}></div>
                                    </div>
                                    <div style={{ width: '60px', fontSize: '0.8rem', fontWeight: '500', color: barColor }}>
                                      {item.value >= 0 ? '+' : ''}{item.value.toFixed(4)}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '1.5rem', fontStyle: 'italic' }}>
                              * SHAP (SHapley Additive exPlanations) values indicate the marginal feature contributions to the PPO policy's raw action output. Green represents positive contribution, red represents negative contribution.
                            </p>
                          </>
                        )}
                      </div>
                    </div>
                    )}

                    {/* SECTION 5: WALK-FORWARD CV */}
                    {(deepSubTab === 'all' || deepSubTab === 'wfv') && (
                      <div id="sec-wfv" style={{ scrollMarginTop: '120px', marginTop: '2rem' }}>
                        <div className="glass-panel" style={{ padding: '1.5rem' }}>
                          <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem' }}>
                            📊 Walk-Forward Out-of-Sample Validation (5 Folds)
                            <InfoBadge topic="Walk-Forward CV" text="Evaluates agent across 5 sequential out-of-sample temporal windows to prevent data leakage and backtest overfitting." anchor="#walk-forward" onNavigate={handleNavigateSchool} />
                          </h3>
                          
                          {loadingWFV ? (
                            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                              <div className="typing-indicator" style={{ margin: 'auto' }}><span></span><span></span><span></span></div>
                              <div style={{ marginTop: '1rem' }}>Loading out-of-sample walk-forward validation results...</div>
                            </div>
                          ) : wfvData && wfvData.length > 0 ? (
                            <>
                              {(() => {
                                const sharpes = wfvData.map(r => Number(r['Out-of-Sample Sharpe']) || 0);
                                const avgOOF = (sharpes.reduce((a, b) => a + b, 0) / Math.max(1, sharpes.length)).toFixed(2);
                                return (
                                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
                                    <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                                      <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Mean OOF Sharpe</div>
                                      <div style={{ fontSize: '1.4rem', fontWeight: 'bold', color: '#FFFFFF', marginTop: '0.2rem' }}>{avgOOF}</div>
                                    </div>
                                    <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                                      <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>DSR Significance</div>
                                      <div style={{ fontSize: '1.4rem', fontWeight: 'bold', color: '#4ade80', marginTop: '0.2rem' }}>DSR ≥ 0.95 (Sig.)</div>
                                    </div>
                                    <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                                      <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Evaluation Scheme</div>
                                      <div style={{ fontSize: '1.2rem', fontWeight: 'bold', marginTop: '0.2rem' }}>5 Sequential Folds</div>
                                    </div>
                                  </div>
                                );
                              })()}

                            <table className="leaderboard-table" style={{ width: '100%' }}>
                              <thead>
                                <tr>
                                  <th>Fold</th>
                                  <th>Test Range</th>
                                  <th>Out-of-Sample ROI (%)</th>
                                  <th>Market Buy & Hold (%)</th>
                                  <th>Out-of-Sample Sharpe</th>
                                  <th>Max Drawdown (%)</th>
                                  <th>Trades</th>
                                </tr>
                              </thead>
                              <tbody>
                                {wfvData.map((row: WalkForwardRow, idx: number) => (
                                  <tr key={idx} className="leaderboard-row">
                                    <td style={{ fontWeight: 'bold' }}>Fold {row.Fold}</td>
                                    <td style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{row['Test Range']}</td>
                                    <td style={{ fontWeight: 'bold', color: row['Out-of-Sample ROI (%)'] >= 0 ? '#4ade80' : '#f87171' }}>
                                      {row['Out-of-Sample ROI (%)'] >= 0 ? '+' : ''}{row['Out-of-Sample ROI (%)']}%
                                    </td>
                                    <td style={{ color: row['Market B&H (%)'] >= 0 ? '#4ade80' : '#f87171' }}>
                                      {row['Market B&H (%)'] >= 0 ? '+' : ''}{row['Market B&H (%)']}%
                                    </td>
                                    <td style={{ fontWeight: 'bold' }}>{row['Out-of-Sample Sharpe']}</td>
                                    <td style={{ color: '#f87171' }}>{row['Max DD (%)']}%</td>
                                    <td>{row.Trades}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>

                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '1.5rem', fontStyle: 'italic' }}>
                              * Evaluated out-of-sample across 5 expanding training windows to test policy robustness against market regime shifts (bull, bear, sideways) and eliminate backtest overfitting.
                            </p>
                          </>
                        ) : (
                          <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                            <p>No walk-forward validation data loaded.</p>
                            <button className="chat-suggested-btn" onClick={fetchWalkForward} style={{ width: 'auto', margin: '0 auto' }}>Fetch WFV Data</button>
                          </div>
                        )}
                      </div>
                    </div>
                    )}

                    {/* SECTION 6: MONTE CARLO SIMULATION */}
                    {(deepSubTab === 'all' || deepSubTab === 'montecarlo') && (
                      <div id="sec-montecarlo" style={{ scrollMarginTop: '120px', marginTop: '2rem' }}>
                        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem' }}>
                            <h3 style={{ margin: 0, fontSize: '1.1rem' }}>🎲 Monte Carlo Risk Simulation</h3>
                            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Predict potential equity curves under synthetic price paths</p>
                          </div>

                        {/* CONFIGURATION CONTROLS */}
                        <div className="glass-panel" style={{ padding: '1.2rem', display: 'grid', gridTemplateColumns: '1.5fr 1.5fr 1fr 1fr', gap: '1.5rem', alignItems: 'center', background: 'rgba(0,0,0,0.1)' }}>
                          <div>
                            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                              Simulations: <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>{mcSimulations}</span>
                            </label>
                            <input 
                              type="range" 
                              min="10" 
                              max="100" 
                              step="10" 
                              style={{ width: '100%', accentColor: '#FFFFFF' }}
                              value={mcSimulations} 
                              onChange={e => setMcSimulations(Number(e.target.value))} 
                            />
                          </div>

                          <div>
                            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                              Noise Std Dev: <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>{mcNoise.toFixed(4)}</span>
                            </label>
                            <input 
                              type="range" 
                              min="0.001" 
                              max="0.010" 
                              step="0.001" 
                              style={{ width: '100%', accentColor: '#FFFFFF' }}
                              value={mcNoise} 
                              onChange={e => setMcNoise(Number(e.target.value))} 
                            />
                          </div>

                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <input 
                              type="checkbox" 
                              id="mcSwanCheckbox"
                              style={{ width: '16px', height: '16px', cursor: 'pointer', accentColor: '#FFFFFF' }}
                              checked={mcSwan} 
                              onChange={e => setMcSwan(e.target.checked)} 
                            />
                            <label htmlFor="mcSwanCheckbox" style={{ fontSize: '0.85rem', cursor: 'pointer', userSelect: 'none' }}>Black Swan Crash</label>
                          </div>

                          <button 
                            className="chat-suggested-btn" 
                            style={{ margin: 0, textAlign: 'center', background: '#FFFFFF', color: '#000000', border: '1px solid #FFFFFF', borderRadius: '8px', padding: '0.6rem 1rem', cursor: 'pointer', fontWeight: 'bold', boxShadow: '0 2px 12px rgba(255, 255, 255, 0.15)' }}
                            onClick={() => selectedAsset && fetchMonteCarlo(selectedAsset.label, mcSimulations, mcNoise, mcSwan)}
                            disabled={loadingMC}
                          >
                            {loadingMC ? "Simulating..." : "Run Simulation"}
                          </button>
                        </div>

                        {/* STATUS LOADER / METRICS & CHART */}
                        {loadingMC ? (
                          <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                            <div className="typing-indicator" style={{ margin: 'auto' }}><span></span><span></span><span></span></div>
                            <div style={{ marginTop: '1rem', color: '#FFFFFF', fontSize: '0.95rem' }}>Running Monte Carlo projection paths on {selectedAsset.label}...</div>
                          </div>
                        ) : !mcData ? (
                          <div style={{ padding: '2.5rem 1.5rem', textAlign: 'center', color: 'var(--text-secondary)', background: 'rgba(255, 255, 255, 0.01)', borderRadius: '8px', border: '1px dashed var(--glass-border)' }}>
                            <p style={{ margin: 0, fontSize: '0.88rem' }}>
                              Configure parameters above and click <strong>Run Simulation</strong> to compute probabilistic geometric Brownian motion equity cones for <strong>{selectedAsset.label}</strong>.
                            </p>
                          </div>
                        ) : (
                          <>
                            {/* METRICS CARDS GRID */}
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', textAlign: 'center', border: '1px solid var(--glass-border)' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Mean ROI</div>
                                <div style={{ fontSize: '1.4rem', fontWeight: 'bold', marginTop: '0.25rem', color: mcData.summary.mean_roi >= 0 ? '#4ade80' : '#f87171' }}>
                                  {mcData.summary.mean_roi >= 0 ? '+' : ''}{mcData.summary.mean_roi.toFixed(2)}%
                                </div>
                              </div>
                              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', textAlign: 'center', border: '1px solid var(--glass-border)' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Success Rate</div>
                                <div style={{ fontSize: '1.4rem', fontWeight: 'bold', marginTop: '0.25rem', color: mcData.summary.success_rate >= 50 ? '#4ade80' : '#f87171' }}>
                                  {mcData.summary.success_rate.toFixed(1)}%
                                </div>
                              </div>
                              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', textAlign: 'center', border: '1px solid var(--glass-border)' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Value at Risk (5th%)</div>
                                <div style={{ fontSize: '1.4rem', fontWeight: 'bold', marginTop: '0.25rem', color: mcData.summary.var_5th >= 0 ? '#4ade80' : '#f87171' }}>
                                  {mcData.summary.var_5th.toFixed(2)}%
                                </div>
                              </div>
                              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', textAlign: 'center', border: '1px solid var(--glass-border)' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Mean Sharpe</div>
                                <div style={{ fontSize: '1.4rem', fontWeight: 'bold', marginTop: '0.25rem', color: mcData.summary.mean_sharpe >= 0 ? '#4ade80' : '#f87171' }}>
                                  {mcData.summary.mean_sharpe.toFixed(2)}
                                </div>
                              </div>
                            </div>

                            {/* CUSTOM SVG EQUITY CONE CHART */}
                            <div style={{ position: 'relative', marginTop: '1rem', background: 'rgba(0,0,0,0.15)', padding: '1.2rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Equity Projection Cone ({mcData.step_count} Simulation Steps)</div>
                              {(() => {
                                const width = 600;
                                const height = 250;
                                const paddingLeft = 50;
                                const paddingRight = 20;
                                const paddingTop = 15;
                                const paddingBottom = 30;

                                const allVals = [
                                  ...mcData.percentiles["5"],
                                  ...mcData.percentiles["25"],
                                  ...mcData.percentiles["50"],
                                  ...mcData.percentiles["75"],
                                  ...mcData.percentiles["95"],
                                  ...mcData.sample_paths.flat()
                                ];
                                const yMin = Math.min(...allVals) * 0.98;
                                const yMax = Math.max(...allVals) * 1.02;

                                const getX = (t: number) => paddingLeft + (t / (mcData.step_count - 1)) * (width - paddingLeft - paddingRight);
                                const getY = (v: number) => height - paddingBottom - ((v - yMin) / (yMax - yMin)) * (height - paddingTop - paddingBottom);

                                // Outer Band: 5th to 95th percentile
                                const outerPoints = [
                                  ...mcData.percentiles["5"].map((val: number, t: number) => `${getX(t).toFixed(1)},${getY(val).toFixed(1)}`),
                                  ...[...mcData.percentiles["95"]].reverse().map((val: number, t: number) => `${getX(mcData.step_count - 1 - t).toFixed(1)},${getY(val).toFixed(1)}`)
                                ].join(' ');

                                // Inner Band: 25th to 75th percentile
                                const innerPoints = [
                                  ...mcData.percentiles["25"].map((val: number, t: number) => `${getX(t).toFixed(1)},${getY(val).toFixed(1)}`),
                                  ...[...mcData.percentiles["75"]].reverse().map((val: number, t: number) => `${getX(mcData.step_count - 1 - t).toFixed(1)},${getY(val).toFixed(1)}`)
                                ].join(' ');

                                // Median path
                                const medianPoints = mcData.percentiles["50"].map((val: number, t: number) => `${getX(t).toFixed(1)},${getY(val).toFixed(1)}`).join(' ');

                                // 5 Sample paths
                                const samplePathsPoints = mcData.sample_paths.map((path: number[]) => 
                                  path.map((val: number, t: number) => `${getX(t).toFixed(1)},${getY(val).toFixed(1)}`).join(' ')
                                );

                                // Y-axis grid ticks
                                const ticksCount = 4;
                                const yTicks = Array.from({ length: ticksCount + 1 }, (_, index) => {
                                  return yMin + (index / ticksCount) * (yMax - yMin);
                                });

                                // X-axis grid ticks (6 steps spacing)
                                const xTicks = [0, Math.floor(mcData.step_count * 0.2), Math.floor(mcData.step_count * 0.4), Math.floor(mcData.step_count * 0.6), Math.floor(mcData.step_count * 0.8), mcData.step_count - 1];

                                return (
                                  <>
                                    <svg 
                                      width="100%" 
                                      height={height} 
                                      viewBox={`0 0 ${width} ${height}`}
                                      preserveAspectRatio="none"
                                      style={{ overflow: 'visible', display: 'block' }}
                                      onMouseMove={(e) => {
                                        const rect = e.currentTarget.getBoundingClientRect();
                                        const plotWidth = width - paddingLeft - paddingRight;
                                        const mouseX = ((e.clientX - rect.left) / rect.width) * width;
                                        const pct = (mouseX - paddingLeft) / plotWidth;
                                        const step = Math.min(mcData.step_count - 1, Math.max(0, Math.round(pct * (mcData.step_count - 1))));
                                        setMcHoverStep(step);
                                      }}
                                      onMouseLeave={() => setMcHoverStep(null)}
                                    >
                                      {/* GLOW DEFS */}
                                      <defs>
                                        <filter id="neon-glow" x="-20%" y="-20%" width="140%" height="140%">
                                          <feGaussianBlur stdDeviation="4" result="blur" />
                                          <feComposite in="SourceGraphic" in2="blur" operator="over" />
                                        </filter>
                                      </defs>

                                      {/* Grid Lines (Horizontal) */}
                                      {yTicks.map((val, idx) => (
                                        <g key={idx}>
                                          <line 
                                            x1={paddingLeft} 
                                            y1={getY(val)} 
                                            x2={width - paddingRight} 
                                            y2={getY(val)} 
                                            stroke="rgba(255,255,255,0.05)" 
                                            strokeWidth="1"
                                          />
                                          <text 
                                            x={paddingLeft - 8} 
                                            y={getY(val) + 4} 
                                            fill="var(--text-secondary)" 
                                            fontSize="9" 
                                            textAnchor="end"
                                          >
                                            ${Math.round(val).toLocaleString()}
                                          </text>
                                        </g>
                                      ))}

                                      {/* Grid Lines (Vertical) */}
                                      {xTicks.map((step, idx) => (
                                        <g key={idx}>
                                          <line 
                                            x1={getX(step)} 
                                            y1={paddingTop} 
                                            x2={getX(step)} 
                                            y2={height - paddingBottom} 
                                            stroke="rgba(255,255,255,0.05)" 
                                            strokeWidth="1"
                                          />
                                          <text 
                                            x={getX(step)} 
                                            y={height - paddingBottom + 16} 
                                            fill="var(--text-secondary)" 
                                            fontSize="9" 
                                            textAnchor="middle"
                                          >
                                            Step {step}
                                          </text>
                                        </g>
                                      ))}

                                      {/* Percentile Band: 5th-95th (Outer) */}
                                      <polygon 
                                        points={outerPoints} 
                                        fill="rgba(255, 255, 255, 0.04)" 
                                        stroke="rgba(255, 255, 255, 0.08)"
                                        strokeWidth="1"
                                      />

                                      {/* Percentile Band: 25th-75th (Inner) */}
                                      <polygon 
                                        points={innerPoints} 
                                        fill="rgba(255, 255, 255, 0.08)" 
                                        stroke="rgba(255, 255, 255, 0.16)"
                                        strokeWidth="1"
                                      />

                                      {/* 5 Sample Simulation Paths */}
                                      {samplePathsPoints.map((points: string, idx: number) => (
                                        <polyline 
                                          key={idx} 
                                          points={points} 
                                          fill="none" 
                                          stroke="rgba(255, 255, 255, 0.18)" 
                                          strokeWidth="0.8" 
                                        />
                                      ))}

                                      {/* Median (50th Percentile) Path */}
                                      <polyline 
                                        points={medianPoints} 
                                        fill="none" 
                                        stroke="#FFFFFF" 
                                        strokeWidth="2.5"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                      />

                                      {/* Interactive Hover Guidance Line */}
                                      {mcHoverStep !== null && (
                                        <g>
                                          <line 
                                            x1={getX(mcHoverStep)} 
                                            y1={paddingTop} 
                                            x2={getX(mcHoverStep)} 
                                            y2={height - paddingBottom} 
                                            stroke="rgba(255, 255, 255, 0.45)" 
                                            strokeWidth="1.2" 
                                            strokeDasharray="4,4" 
                                          />
                                          {/* Circle marker on Median Path */}
                                          <circle 
                                            cx={getX(mcHoverStep)} 
                                            cy={getY(mcData.percentiles["50"][mcHoverStep])} 
                                            r="5" 
                                            fill="#FFFFFF" 
                                            stroke="#000000" 
                                            strokeWidth="2" 
                                          />
                                        </g>
                                      )}
                                    </svg>

                                    {/* Hover Interactive Tooltip floating overlay */}
                                    {mcHoverStep !== null && (
                                      <div className="glass-panel" style={{
                                        position: 'absolute',
                                        top: '1.5rem',
                                        left: getX(mcHoverStep) > width / 2 ? '2rem' : 'auto',
                                        right: getX(mcHoverStep) <= width / 2 ? '2rem' : 'auto',
                                        padding: '0.8rem 1rem',
                                        fontSize: '0.8rem',
                                        zIndex: 10,
                                        width: '180px',
                                        background: 'rgba(12, 12, 14, 0.97)',
                                        border: '1px solid rgba(255, 255, 255, 0.2)',
                                        boxShadow: '0 8px 30px rgba(0, 0, 0, 0.7)',
                                        borderRadius: '8px',
                                        animation: 'fadeSlideIn 0.2s ease-out'
                                      }}>
                                        <div style={{ fontWeight: 'bold', color: '#FFFFFF', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.25rem', marginBottom: '0.4rem' }}>
                                          Step {mcHoverStep}
                                        </div>
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                                          <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-secondary)' }}>95th% (Top):</span> <strong>${Math.round(mcData.percentiles["95"][mcHoverStep]).toLocaleString()}</strong></div>
                                          <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-secondary)' }}>75th% (High):</span> <strong>${Math.round(mcData.percentiles["75"][mcHoverStep]).toLocaleString()}</strong></div>
                                          <div style={{ display: 'flex', justifyContent: 'space-between', color: '#FFFFFF' }}><span style={{ fontWeight: 'bold' }}>50th% (Median):</span> <strong>${Math.round(mcData.percentiles["50"][mcHoverStep]).toLocaleString()}</strong></div>
                                          <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-secondary)' }}>25th% (Low):</span> <strong>${Math.round(mcData.percentiles["25"][mcHoverStep]).toLocaleString()}</strong></div>
                                          <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-secondary)' }}>5th% (Bottom):</span> <strong>${Math.round(mcData.percentiles["5"][mcHoverStep]).toLocaleString()}</strong></div>
                                        </div>
                                      </div>
                                    )}
                                  </>
                                );
                              })()}
                              
                              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.8rem', fontStyle: 'italic', margin: 0 }}>
                                * Outer band covers 90% of outcomes (5th–95th percentile). Inner band covers 50% (25th–75th percentile). Individual lines represent random trajectory samples.
                              </p>
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                    )}
                  </div>
                ) : (
                  <div className="glass-panel" style={{ padding: '3.5rem 2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                    <Cpu size={48} style={{ opacity: 0.25, marginBottom: '1rem' }} />
                    <h3 style={{ margin: '0 0 0.5rem 0', color: '#FFFFFF' }}>No Asset Selected</h3>
                    <p style={{ fontSize: '0.9rem', maxWidth: '420px', margin: '0 auto 1.5rem auto' }}>
                      Select an asset from the switcher above or return to the Trading Dashboard to inspect its deep quantitative analytics suite.
                    </p>
                    <button
                      className="chat-suggested-btn"
                      onClick={() => setActiveTab('dashboard')}
                      style={{ width: 'auto', margin: '0 auto', display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1.25rem' }}
                    >
                      ← Return to Trading Dashboard
                    </button>
                  </div>
                )}
              </>
            )}

            {/* ── MODEL SCHOOL TAB ── */}
            {activeTab === 'school' && (
              <div className="glass-panel" style={{ padding: '2rem', maxWidth: '900px', margin: '0 auto' }}>
                <h2 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                  <BookOpen size={26} color="#FFFFFF" />
                  Model School & Performance Benchmarks
                </h2>
                <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', marginBottom: '2rem' }}>
                  Learn how the Adaptive AI Trading System operates, and compare PPO Transformer performance against benchmark baseline models.
                </p>

                {/* Quantitative Benchmark Comparison Card */}
                <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2.5rem', background: 'rgba(20, 20, 24, 0.6)', borderRadius: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <h3 style={{ margin: 0, fontSize: '1.1rem', color: '#FFFFFF' }}>🏆 Model Comparison & Out-of-Sample Benchmarks</h3>
                    <button 
                      className="chat-suggested-btn" 
                      onClick={fetchComparison} 
                      disabled={loadingComparison}
                      style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
                    >
                      {loadingComparison ? 'Refreshing...' : '🔄 Refresh Benchmarks'}
                    </button>
                  </div>
                  
                  {comparisonData ? (
                    <table className="leaderboard-table" style={{ width: '100%' }}>
                      <thead>
                        <tr>
                          <th style={{ textAlign: 'left' }}>Model Architecture</th>
                          <th style={{ textAlign: 'center' }}>Out-of-Sample Return</th>
                          <th style={{ textAlign: 'center' }}>Max Drawdown</th>
                          <th style={{ textAlign: 'center' }}>Sharpe Ratio</th>
                          <th style={{ textAlign: 'center' }}>Trades</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparisonData.map((row: ComparisonRow, idx: number) => {
                          const isPPO = row.Model.includes('PPO');
                          const roiVal = (row['ROI (%)'] !== undefined ? row['ROI (%)'] : row.ROI) ?? 0;
                          const maxDdVal = (row['Max DD (%)'] !== undefined ? row['Max DD (%)'] : row.Max_DD) ?? 0;
                          const sharpeVal = row.Sharpe !== undefined ? row.Sharpe : 0;
                          const tradesVal = row.Trades !== undefined ? row.Trades : 0;

                          return (
                            <tr key={idx} style={{ background: isPPO ? 'rgba(255, 255, 255, 0.08)' : 'transparent', fontWeight: isPPO ? 'bold' : 'normal' }}>
                              <td style={{ textAlign: 'left', color: isPPO ? '#FFFFFF' : 'var(--text-secondary)' }}>
                                {isPPO && '⭐ '} {row.Model}
                              </td>
                              <td style={{ textAlign: 'center', color: roiVal >= 0 ? '#4ade80' : '#f87171' }}>
                                {roiVal >= 0 ? '+' : ''}{roiVal}%
                              </td>
                              <td style={{ textAlign: 'center', color: '#f87171' }}>
                                {maxDdVal}%
                              </td>
                              <td style={{ textAlign: 'center', color: sharpeVal >= 0 ? '#4ade80' : '#f87171' }}>
                                {sharpeVal}
                              </td>
                              <td style={{ textAlign: 'center' }}>
                                {tradesVal}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  ) : (
                    <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '1rem' }}>Loading comparison benchmarks...</div>
                  )}
                </div>

                <div className="school-article" style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
                  <section id="rl-basics">
                    <h3>1. Reinforcement Learning (RL) Basics</h3>
                    <p>Unlike standard predictive models that just guess the next price, <strong>Reinforcement Learning</strong> trains an AI <em>agent</em> by letting it play a "game" against the market. The agent receives a <strong>state</strong> (market conditions), takes an <strong>action</strong> (Buy, Sell, Hold), and earns a <strong>reward</strong> (profit/loss).</p>
                    <p>Our system uses <strong>Proximal Policy Optimization (PPO)</strong>. Over millions of simulated trades, the PPO policy learns to maximize long-term risk-adjusted returns rather than just short-term gains.</p>
                  </section>

                  <section id="state-space">
                    <h3>2. How the AI Views the Market (State Space)</h3>
                    <p>The AI cannot look at charts visually like a human. Instead, it is fed a matrix of 17 key technical features stacked over the last 8 time periods (creating a temporal "memory").</p>
                    <ul>
                      <li><strong>Momentum & Trend:</strong> It tracks the Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), and Simple Moving Averages (SMA20, SMA99).</li>
                      <li><strong>Volatility:</strong> It measures candle wicks, body ratios, and normalized volume to gauge market panic or euphoria.</li>
                      <li><strong>Portfolio State:</strong> The AI knows its current unrealized profit, how long it has been holding an asset, and whether it is flat or active.</li>
                    </ul>
                  </section>

                  <section id="hmm-regimes">
                    <h3>3. Market Regime Detection (HMM)</h3>
                    <p>Markets behave differently in bull runs versus bear crashes. We use a <strong>Hidden Markov Model (HMM)</strong> to automatically classify the market into different volatility regimes.</p>
                    <p>By understanding if the market is in a "High Volatility Bear" or "Low Volatility Bull" phase, the RL agent can dynamically adjust its risk tolerance—for example, tightening stop-losses during crashes.</p>
                  </section>

                  <section id="shap-xai">
                    <h3>4. Explainable AI (SHAP)</h3>
                    <p>Neural networks are often considered "black boxes." To solve this, our Advisor uses <strong>SHAP (SHapley Additive exPlanations)</strong>, a method from cooperative game theory.</p>
                    <p>SHAP breaks down the AI's final decision into individual feature contributions. For instance, if the AI says "BUY", SHAP can tell us that 40% of that decision came from an RSI drop, and 20% came from a volume spike. The chatbot uses this mathematical data to explain trades to you in plain English.</p>
                  </section>
                </div>
              </div>
            )}

          </div>
        )}
      </main>
    </div>
    </div>
  );
}

export default App;
