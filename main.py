from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple
from pipeline.data import load_csv, validate_regular_intervals, impute_and_clean, split_train_analysis
from pipeline.model import AnomalyEnsemble
from pipeline.scoring import rank_normalize, percentile_scale, calibrate_training, smooth_series
from pipeline.io import add_output_columns

def choose_top_features(
    colnames: List[str],
    contrib_vec: np.ndarray,
    min_pct: float = 0.01
) -> List[str]:
    total = float(np.sum(contrib_vec))
    if total <= 0 or not np.isfinite(total):
        return []
    pct = contrib_vec / (total + 1e-12)
    # filter by 1%
    keep_idx = np.where(pct >= min_pct)[0]
    entries = []
    for i in keep_idx:
        entries.append((float(contrib_vec[i]), colnames[i]))
    # sort by magnitude desc, tie-break alphabetically
    entries.sort(key=lambda x: (-x[0], x[1]))
    names = [name for _, name in entries][:7]
    # pad with '' to 7 handled by IO
    return names

def generate_report(
    scores: np.ndarray,
    train_mask: np.ndarray,
    top_features: List[List[str]],
    output_path: str,
    train_start: str,
    train_end: str,
    analysis_end: str
) -> None:
    """Generate a comprehensive PDF report for the anomaly detection results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("[WARN] matplotlib/seaborn not available. Skipping PDF report generation.")
        return
    
    # Calculate statistics
    train_scores = scores[train_mask]
    analysis_scores = scores
    
    # Feature frequency analysis
    all_features = []
    for row in top_features:
        all_features.extend([f for f in row if f])
    
    feature_counts = pd.Series(all_features).value_counts()
    
    # Create PDF report
    with PdfPages(output_path) as pdf:
        # Page 1: Overview
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Honeywell Anomaly Detection Report', fontsize=16, fontweight='bold')
        
        # Training period statistics
        axes[0, 0].hist(train_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 0].set_title(f'Training Period Scores\n({train_start} to {train_end})')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(train_scores), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(train_scores):.2f}')
        axes[0, 0].axvline(np.max(train_scores), color='orange', linestyle='--', 
                           label=f'Max: {np.max(train_scores):.2f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Full analysis period scores
        axes[0, 1].hist(analysis_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_title(f'Full Analysis Period Scores\n({train_start} to {analysis_end})')
        axes[0, 1].set_xlabel('Anomaly Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Severity distribution
        severity_bins = [0, 10, 30, 60, 90, 100]
        severity_labels = ['Normal (0-10)', 'Slight (11-30)', 'Moderate (31-60)', 
                          'Significant (61-90)', 'Severe (91-100)']
        severity_counts = []
        for i in range(len(severity_bins)-1):
            count = np.sum((analysis_scores >= severity_bins[i]) & (analysis_scores < severity_bins[i+1]))
            severity_counts.append(count)
        
        axes[1, 0].bar(severity_labels, severity_counts, color=['green', 'yellow', 'orange', 'red', 'darkred'])
        axes[1, 0].set_title('Severity Distribution')
        axes[1, 0].set_ylabel('Number of Rows')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Top contributing features
        if not feature_counts.empty:
            top_10_features = feature_counts.head(10)
            axes[1, 1].barh(range(len(top_10_features)), top_10_features.values)
            axes[1, 1].set_yticks(range(len(top_10_features)))
            axes[1, 1].set_yticklabels(top_10_features.index)
            axes[1, 1].set_title('Top 10 Contributing Features')
            axes[1, 1].set_xlabel('Contribution Count')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No feature contributions\nabove 1% threshold', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Top Contributing Features')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Training validation
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Training Period Validation', fontsize=14, fontweight='bold')
        
        # Training scores over time
        train_times = np.arange(len(train_scores))
        axes[0].plot(train_times, train_scores, 'g-', linewidth=1)
        axes[0].axhline(10, color='orange', linestyle='--', label='Mean threshold (10)')
        axes[0].axhline(25, color='red', linestyle='--', label='Max threshold (25)')
        axes[0].set_title('Training Scores Over Time')
        axes[0].set_xlabel('Time Index')
        axes[0].set_ylabel('Anomaly Score')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Training statistics table
        stats_text = f"""Training Period Statistics:
        
Start: {train_start}
End: {train_end}
Duration: {len(train_scores)} time points

Scores:
• Mean: {np.mean(train_scores):.2f} {'✅' if np.mean(train_scores) < 10 else '❌'}
• Max: {np.max(train_scores):.2f} {'✅' if np.max(train_scores) < 25 else '❌'}
• Std: {np.std(train_scores):.2f}
• Min: {np.min(train_scores):.2f}

Requirements Met: {'✅' if np.mean(train_scores) < 10 and np.max(train_scores) < 25 else '❌'}"""
        
        axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes, 
                     fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1].set_title('Validation Results')
        axes[1].axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    print(f"Generated comprehensive report: {output_path}")

def main():
    ap = argparse.ArgumentParser(description="Honeywell Multivariate Anomaly Detection")
    ap.add_argument('--input_csv', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--time_col', default='Time')
    ap.add_argument('--train_start', required=True)
    ap.add_argument('--train_end', required=True)
    ap.add_argument('--analysis_end', required=True)
    ap.add_argument('--smooth', type=int, default=1, help='median smoothing window for final scores (1 = off)')
    ap.add_argument('--save-report', action='store_true', help='generate comprehensive PDF report')
    args = ap.parse_args()

    # Load
    raw = pd.read_csv(args.input_csv)
    if args.time_col not in raw.columns:
        # try case-insensitive
        matches = [c for c in raw.columns if c.lower() == args.time_col.lower()]
        if matches:
            args.time_col = matches[0]
        else:
            raise ValueError(f"Timestamp column '{args.time_col}' not found.")

    # Keep original for output merging
    original = raw.copy()

    # Prepare numeric frame with datetime index
    df_num = load_csv(args.input_csv, args.time_col)
    validate_regular_intervals(df_num)
    df_num = impute_and_clean(df_num)

    # Align original to cleaned index
    original[args.time_col] = pd.to_datetime(original[args.time_col], errors='coerce')
    original = original.set_index(original[args.time_col]).drop(columns=[args.time_col])
    original = original.reindex(df_num.index)  # align rows

    # Split
    train_df, analysis_df = split_train_analysis(df_num, args.train_start, args.train_end, args.analysis_end)

    # Fit model
    model = AnomalyEnsemble()
    model.fit(train_df)

    # Score all analysis rows
    ch = model.score_rows(analysis_df)
    S1, S2, S3 = ch['S1'], ch['S2'], ch['S3']

    # Rank-normalize channels to [0,1] then average
    R1 = rank_normalize(S1)
    R2 = rank_normalize(S2)
    R3 = rank_normalize(S3)
    combined = (R1 + R2 + R3) / 3.0

    # Map to 0-100 via percentiles over analysis window
    scaled = percentile_scale(combined)

    # Build training mask over analysis index
    idx = analysis_df.index
    train_mask = (idx >= pd.to_datetime(args.train_start)) & (idx <= pd.to_datetime(args.train_end))

    # Calibrate to meet training constraints
    scaled = calibrate_training(scaled, train_mask)

    # Optional smoothing to avoid spiky jumps
    if args.smooth and args.smooth > 1:
        scaled = smooth_series(scaled, window=int(args.smooth))

    # Row-wise feature attributions
    top_lists: List[List[str]] = []
    Z = ch['Z']
    RE = ch['RECON']

    # Base contributions (z + PCA reconstruction) for all rows
    z_abs = np.abs(Z)              # (n_rows, n_feats)
    re_abs = np.abs(RE)            # (n_rows, n_feats)

    # Fast batched IsolationForest sensitivity: one pass per feature
    if_delta = model.if_sensitivity_matrix(Z)  # (n_rows, n_feats)

    # Total contribution matrix
    total_contrib = z_abs + re_abs + if_delta  # (n_rows, n_feats)

    # Build top features per row
    for r in range(total_contrib.shape[0]):
        contrib = np.maximum(total_contrib[r, :], 0.0)
        top = choose_top_features(model.colnames, contrib, min_pct=0.01)
        top_lists.append(top if len(top) > 0 else [''] * 7)

    # Merge back to original-sized DataFrame over analysis index only
    out_df = original.loc[idx].copy()
    out_df = add_output_columns(out_df, pd.Series(scaled, index=idx, name='Abnormality_score'), top_lists)

    # For rows outside analysis window (if any exist in original), leave as-is
    # Reinsert time column for readability (optional)
    out_df.insert(0, 'Time', out_df.index.astype('datetime64[ns]'))

    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote: {args.output_csv}")
    
    # Generate comprehensive report if requested
    if args.save_report:
        report_path = args.output_csv.replace('.csv', '_Report.pdf')
        generate_report(scaled, train_mask, top_lists, report_path, 
                       args.train_start, args.train_end, args.analysis_end)

if __name__ == '__main__':
    main()
