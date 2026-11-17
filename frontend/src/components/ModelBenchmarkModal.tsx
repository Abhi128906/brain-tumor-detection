import React, { useEffect, useState } from 'react';
import { BenchmarkResponse } from '../types/analysis';
import { fetchModelBenchmark } from '../services/api';
import { X, Cpu, Zap, Database, Layers, ArrowUpRight, Loader2 } from 'lucide-react';

interface ModelBenchmarkModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ModelBenchmarkModal: React.FC<ModelBenchmarkModalProps> = ({ isOpen, onClose }) => {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      fetchModelBenchmark()
        .then(res => {
          setData(res);
          setLoading(false);
        })
        .catch(err => {
          console.error(err);
          setLoading(false);
        });
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md">
      <div className="glass-panel w-full max-w-3xl rounded-2xl border border-slate-700 bg-slate-900 shadow-2xl p-6 relative max-h-[90vh] overflow-y-auto">
        
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
        >
          <X className="h-5 w-5" />
        </button>

        {/* Title */}
        <div className="flex items-center space-x-3 mb-4">
          <div className="h-10 w-10 rounded-xl bg-cyan-600/20 text-cyan-400 flex items-center justify-center">
            <Cpu className="h-6 w-6" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Model Architecture Benchmark</h3>
            <p className="text-xs text-slate-400">
              Comparative latency, memory, parameter scaling & Dice accuracy on held-out evaluation scans.
            </p>
          </div>
        </div>

        {loading ? (
          <div className="py-16 flex flex-col items-center justify-center space-y-3">
            <Loader2 className="h-8 w-8 text-cyan-400 animate-spin" />
            <span className="text-sm text-slate-400">Running inference benchmarks on {data?.device_used || 'system hardware'}...</span>
          </div>
        ) : (
          <div className="space-y-6">
            
            {/* Speedup Highlight */}
            {data && (
              <div className="p-4 rounded-xl bg-cyan-950/40 border border-cyan-800/60 flex items-center justify-between">
                <div>
                  <span className="text-xs text-cyan-300 font-semibold uppercase tracking-wider block">Edge Acceleration Ratio</span>
                  <span className="text-2xl font-bold text-white font-mono mt-0.5 block">
                    {data.speedup}x Faster Inference
                  </span>
                  <span className="text-xs text-slate-400">Benchmarked on {data.device_used}</span>
                </div>
                <div className="h-12 w-12 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center">
                  <Zap className="h-6 w-6" />
                </div>
              </div>
            )}

            {/* Comparison Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {data?.models.map((model, idx) => (
                <div 
                  key={idx}
                  className={`p-5 rounded-xl border flex flex-col justify-between ${
                    idx === 0 
                      ? 'bg-slate-950/70 border-slate-800' 
                      : 'bg-cyan-950/20 border-cyan-800/60 shadow-lg shadow-cyan-950/30'
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono">
                        {idx === 0 ? 'Full Precision Reference' : 'Pruned Edge Compact'}
                      </span>
                      <span className="text-xs text-cyan-400 font-mono font-bold">{model.compression_ratio}</span>
                    </div>

                    <h4 className="text-base font-bold text-white mt-2.5">{model.name}</h4>
                    <p className="text-xs text-slate-400 mt-1 mb-2">{model.intended_deployment}</p>
                    {model.weights_trained === false && (
                      <p className="text-[11px] text-amber-300 mb-2">Weights not trained yet - latency/size are for the architecture only.</p>
                    )}

                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between py-1 border-b border-slate-800">
                        <span className="text-slate-400">Parameter Count:</span>
                        <span className="font-mono text-white font-semibold">{model.parameters} ({model.param_count.toLocaleString()})</span>
                      </div>
                      <div className="flex justify-between py-1 border-b border-slate-800">
                        <span className="text-slate-400">Median Latency:</span>
                        <span className="font-mono text-cyan-300 font-semibold">{model.latency_ms} ms</span>
                      </div>
                      <div className="flex justify-between py-1 border-b border-slate-800">
                        <span className="text-slate-400">Parameter Memory:</span>
                        <span className="font-mono text-slate-200">{model.memory_mb} MB</span>
                      </div>
                      <div className="flex justify-between py-1 border-b border-slate-800">
                        <span className="text-slate-400">Test Dice Score:</span>
                        <span className="font-mono text-emerald-400 font-bold">{model.val_dice_score ?? 'not evaluated'}</span>
                      </div>
                      <div className="flex justify-between py-1">
                        <span className="text-slate-400">Test IoU:</span>
                        <span className="font-mono text-slate-200">{model.val_iou ?? 'not evaluated'}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="text-[11px] text-slate-400 p-3 rounded-lg bg-slate-950 border border-slate-800 leading-relaxed">
              <strong>Evaluation Protocol: </strong>
              Latency, parameter count and parameter memory are measured live on this machine. Dice / IoU are read from models/segmentation_metrics.json, produced by training/train_segmentation.py on the held-out test split, and show "not evaluated" until that script has been run.
            </div>

          </div>
        )}

      </div>
    </div>
  );
};
