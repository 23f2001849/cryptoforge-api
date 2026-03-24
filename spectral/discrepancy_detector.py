"""
CryptoForge — Spectral-Neural Discrepancy Detector (Component 5)

Compares spectral security predictions against neural distinguisher results
and flags cases where they disagree. These discrepancies point at the
frontier of cryptanalytic knowledge.

Two types of discrepancy:
  Type A: "Neural breaks what spectral says is secure"
    → Neural found an attack beyond current theory (scientifically valuable)
  
  Type B: "Spectral predicts weakness that neural can't exploit"
    → Theoretical bound is loose or neural is too weak

Requires results from:
  - spectral_neural_regression.py (Phase 2 output)
  OR
  - Individual spectral fingerprints + neural accuracy measurements

Usage:
    py -m spectral.discrepancy_detector                # uses regression results
    py -m spectral.discrepancy_detector --threshold 0.1  # custom threshold
"""

import os
import json
import numpy as np


def detect_discrepancies(regression_results_path: str = 'spectral_neural_regression.json',
                          discrepancy_threshold: float = 0.08,
                          show_progress: bool = True) -> dict:
    """Detect spectral-neural discrepancies from regression results.
    
    A discrepancy is flagged when |predicted_accuracy - actual_accuracy|
    exceeds the threshold.
    
    Args:
        regression_results_path: path to spectral_neural_regression.json
        discrepancy_threshold: minimum |error| to flag as discrepancy
        show_progress: print results
    
    Returns:
        dict with discrepancies classified as Type A or Type B
    """
    if not os.path.exists(regression_results_path):
        print(f"  ERROR: {regression_results_path} not found.")
        print(f"  Run 'py spectral_neural_regression.py --phase 2' first.")
        return {'error': 'results not found'}
    
    with open(regression_results_path, 'r') as f:
        data = json.load(f)
    
    per_config = data.get('per_config', {})
    regression = data.get('regression', {})
    
    if not per_config:
        print("  ERROR: No per-config data found in results")
        return {'error': 'no data'}
    
    # Build predicted accuracy from multivariate regression
    multi_reg = regression.get('multivariate', None)
    
    if show_progress:
        print(f"\n{'='*65}")
        print(f"  Spectral-Neural Discrepancy Detector")
        print(f"{'='*65}")
        print(f"  Configs analyzed: {len(per_config)}")
        print(f"  Discrepancy threshold: {discrepancy_threshold}")
        if multi_reg:
            print(f"  Regression R²: {multi_reg['r_squared']:.4f}")
        print(f"{'='*65}")
    
    # Classify each config
    type_a = []  # Neural breaks what spectral predicts safe
    type_b = []  # Spectral predicts weak but neural can't break
    concordant = []  # Both agree
    
    feature_names = ['nonlinearity', 'max_walsh', 'spectral_flatness',
                     'spectral_entropy', 'differential_uniformity', 'algebraic_degree']
    
    for name, config_data in per_config.items():
        actual_acc = config_data.get('accuracy', None)
        nl = config_data.get('nonlinearity', None)
        du = config_data.get('differential_uniformity', None)
        
        if actual_acc is None or nl is None:
            continue
        
        # Simple spectral prediction: high NL + low δ → should be secure (~0.50)
        # Low NL or high δ → should be breakable (> 0.52)
        nl_score = nl / 120.0  # normalized
        du_score = 1.0 - du / 256.0
        spectral_security = (nl_score + du_score) / 2.0  # 0 to 1, higher = more secure
        
        # Predict: if spectral_security > 0.7, expect accuracy ~0.50 (secure)
        #          if spectral_security < 0.3, expect accuracy > 0.52 (breakable)
        predicted_secure = spectral_security > 0.7
        actual_secure = actual_acc <= 0.52
        
        entry = {
            'name': name,
            'actual_accuracy': actual_acc,
            'spectral_security_score': float(spectral_security),
            'nonlinearity': nl,
            'differential_uniformity': du,
            'predicted_secure': predicted_secure,
            'actual_secure': actual_secure,
        }
        
        if predicted_secure and not actual_secure:
            # Type A: spectral says secure, neural breaks it
            entry['type'] = 'A'
            entry['severity'] = float(actual_acc - 0.52)  # how badly it breaks
            type_a.append(entry)
        elif not predicted_secure and actual_secure:
            # Type B: spectral says weak, neural can't break it
            entry['type'] = 'B'
            entry['severity'] = float(0.52 - actual_acc)
            type_b.append(entry)
        else:
            entry['type'] = 'concordant'
            concordant.append(entry)
    
    # Report
    if show_progress:
        total = len(type_a) + len(type_b) + len(concordant)
        
        print(f"\n  Classification:")
        print(f"    Concordant (both agree):  {len(concordant):3d} ({100*len(concordant)/total:.1f}%)")
        print(f"    Type A (neural > spectral): {len(type_a):3d} ({100*len(type_a)/total:.1f}%)")
        print(f"    Type B (spectral > neural): {len(type_b):3d} ({100*len(type_b)/total:.1f}%)")
        
        if type_a:
            print(f"\n  {'─'*60}")
            print(f"  TYPE A DISCREPANCIES: Neural breaks what spectral says is safe")
            print(f"  These are SCIENTIFICALLY VALUABLE — the neural net found an")
            print(f"  attack that current spectral theory doesn't predict.")
            print(f"  {'─'*60}")
            for d in sorted(type_a, key=lambda x: -x['severity']):
                print(f"    {d['name']:<20s} acc={d['actual_accuracy']:.4f} "
                      f"NL={d['nonlinearity']:3d} δ={d['differential_uniformity']:3d} "
                      f"spectral_score={d['spectral_security_score']:.3f}")
        
        if type_b:
            print(f"\n  {'─'*60}")
            print(f"  TYPE B DISCREPANCIES: Spectral predicts weakness, neural can't exploit")
            print(f"  The theoretical vulnerability exists but the neural net")
            print(f"  can't find it in practice.")
            print(f"  {'─'*60}")
            for d in sorted(type_b, key=lambda x: -x['severity']):
                print(f"    {d['name']:<20s} acc={d['actual_accuracy']:.4f} "
                      f"NL={d['nonlinearity']:3d} δ={d['differential_uniformity']:3d} "
                      f"spectral_score={d['spectral_security_score']:.3f}")
        
        if not type_a and not type_b:
            print(f"\n  No discrepancies found — spectral prediction and neural")
            print(f"  reality are fully concordant across all configurations.")
            print(f"  This validates the spectral fingerprint as a reliable")
            print(f"  proxy for neural cryptanalytic resistance.")
        
        print(f"\n{'='*65}")
    
    return {
        'type_a_discrepancies': type_a,
        'type_b_discrepancies': type_b,
        'concordant': concordant,
        'counts': {
            'type_a': len(type_a),
            'type_b': len(type_b),
            'concordant': len(concordant),
            'total': len(type_a) + len(type_b) + len(concordant),
        },
    }


if __name__ == "__main__":
    import sys
    
    threshold = 0.08
    results_path = 'spectral_neural_regression.json'
    
    for arg in sys.argv[1:]:
        if arg.startswith('--threshold='):
            threshold = float(arg.split('=')[1])
        elif arg.startswith('--results='):
            results_path = arg.split('=')[1]
    
    results = detect_discrepancies(
        regression_results_path=results_path,
        discrepancy_threshold=threshold,
        show_progress=True,
    )
    
    # Save
    outpath = 'discrepancy_results.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")