#!/usr/bin/env python3
"""
Test runner script for Streamlit Search App

This script provides a convenient way to run tests with different options.
"""

import sys
import subprocess
import argparse


def run_tests(args):
    """Run pytest with the specified arguments"""
    cmd = ['python', '-m', 'pytest']
    
    if args.verbose:
        cmd.append('-v')
    
    if args.coverage:
        cmd.extend(['--cov=.', '--cov-report=html', '--cov-report=term'])
    
    if args.file:
        cmd.append(f'tests/{args.file}')
    else:
        cmd.append('tests/')
    
    if args.pattern:
        cmd.extend(['-k', args.pattern])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run tests for Streamlit Search App')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Verbose output')
    parser.add_argument('-c', '--coverage', action='store_true',
                       help='Run with coverage reporting')
    parser.add_argument('-f', '--file', type=str,
                       help='Run specific test file (e.g., test_webapp.py)')
    parser.add_argument('-k', '--pattern', type=str,
                       help='Run tests matching pattern')
    
    args = parser.parse_args()
    
    return run_tests(args)


if __name__ == '__main__':
    sys.exit(main())