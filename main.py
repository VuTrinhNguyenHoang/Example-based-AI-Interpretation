#!/usr/bin/env python3
"""
Main script to run all Example-based AI Interpretation experiments
"""

import argparse
import os
import sys
import subprocess
import time
from datetime import datetime

def run_experiment(script_path, args_list, experiment_name):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Starting {experiment_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        cmd = [sys.executable, script_path] + args_list
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print(f"✅ {experiment_name} completed successfully")
        if result.stdout:
            print("Output:")
            print(result.stdout)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ {experiment_name} failed")
        print(f"Error code: {e.returncode}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Duration: {duration:.2f} seconds")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run Example-based AI Interpretation Experiments')
    parser.add_argument('--experiment', type=str, choices=['all', 'classification', 'regression', 'cnn', 'bert'],
                       default='all', help='Which experiment to run')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save all results (Note: Individual experiments use fixed output directory)')
    
    args = parser.parse_args()
    
    # Note: Output directory argument is kept for backward compatibility, 
    # but individual experiments use their own fixed output directories
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(script_dir, 'experiments')
    
    results = {}
    start_time = datetime.now()
    
    print(f"Starting experiments at {start_time}")
    print("Each experiment uses its own fixed parameters and saves to 'results/' directory")
    
    # Classification Experiment
    if args.experiment in ['all', 'classification']:        
        success = run_experiment(
            os.path.join(experiments_dir, 'classification_experiment.py'),
            [],  # No arguments - uses fixed parameters
            'Classification Experiment'
        )
        results['classification'] = success
    
    # Regression Experiment
    if args.experiment in ['all', 'regression']:        
        success = run_experiment(
            os.path.join(experiments_dir, 'regression_experiment.py'),
            [],  # No arguments - uses fixed parameters
            'Regression Experiment'
        )
        results['regression'] = success
    
    # CNN Experiment
    if args.experiment in ['all', 'cnn']:        
        success = run_experiment(
            os.path.join(experiments_dir, 'cnn_experiment.py'),
            [],  # No arguments - uses fixed parameters
            'CNN Experiment'
        )
        results['cnn'] = success
    
    # BERT Experiment
    if args.experiment in ['all', 'bert']:        
        success = run_experiment(
            os.path.join(experiments_dir, 'bert_experiment.py'),
            [],  # No arguments - uses fixed parameters
            'BERT Experiment'
        )
        results['bert'] = success
    
    # Summary
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total duration: {total_duration}")
    print("Output directory: results/")
    
    print("\nResults:")
    for experiment, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {experiment}: {status}")
    
    successful_experiments = sum(results.values())
    total_experiments = len(results)
    
    print(f"\nOverall: {successful_experiments}/{total_experiments} experiments completed successfully")
    
    if successful_experiments == total_experiments:
        print("\n🎉 All experiments completed successfully!")
        print("Check the results in: results/")
    else:
        print(f"\n⚠️  {total_experiments - successful_experiments} experiment(s) failed.")
        print("Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()