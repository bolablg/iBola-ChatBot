#!/usr/bin/env python3
"""
Comprehensive test runner for the chatbot system.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


class TestRunner:
    """Comprehensive test runner with multiple test suites."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = {}

    def run_command(self, command, description):
        """Run a command and capture results."""
        print(f"\n🚀 Running: {description}")
        print("=" * 60)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            success = result.returncode == 0
            output = result.stdout
            errors = result.stderr

            if success:
                print("✅ PASSED"            else:
                print("❌ FAILED"                if errors:
                    print(f"Error: {errors}")

            return {
                'success': success,
                'output': output,
                'errors': errors,
                'return_code': result.returncode
            }

        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT - Test took too long")
            return {'success': False, 'output': '', 'errors': 'Timeout', 'return_code': -1}
        except Exception as e:
            print(f"💥 ERROR: {e}")
            return {'success': False, 'output': '', 'errors': str(e), 'return_code': -1}

    def run_linting(self):
        """Run code linting checks."""
        commands = [
            ("black --check --diff app/ tests/", "Code formatting check (Black)"),
            ("isort --check-only --diff app/ tests/", "Import sorting check (isort)"),
            ("flake8 app/ tests/ --max-line-length=100 --extend-ignore=E203,W503", "Static analysis (flake8)")
        ]

        results = {}
        for command, description in commands:
            results[description] = self.run_command(command, description)

        return results

    def run_unit_tests(self):
        """Run unit tests."""
        command = "pytest tests/test_agents.py tests/test_services.py -v --tb=short --cov=app --cov-report=term-missing"
        return self.run_command(command, "Unit Tests")

    def run_integration_tests(self):
        """Run integration tests."""
        command = "pytest tests/test_integration.py -v --tb=short --cov-append --cov-report=term-missing"
        return self.run_command(command, "Integration Tests")

    def run_security_tests(self):
        """Run security tests."""
        command = "pytest tests/test_security.py -v --tb=short"
        return self.run_command(command, "Security Tests")

    def run_performance_tests(self):
        """Run performance tests."""
        command = "pytest tests/test_performance.py -v --tb=short"
        return self.run_command(command, "Performance Tests")

    def run_all_tests(self):
        """Run all test suites."""
        command = "pytest tests/ -v --tb=short --cov=app --cov-report=term-missing --cov-report=html"
        return self.run_command(command, "All Tests")

    def generate_report(self, results):
        """Generate a comprehensive test report."""
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 80)

        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        for test_name, result in results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            print(f"\n{status}: {test_name}")

            if not result['success']:
                if result['errors']:
                    print(f"   Error: {result['errors'][:200]}...")

            # Count results
            if "Tests" in test_name:
                # This is a rough approximation - in a real scenario you'd parse pytest output
                if result['success']:
                    passed_tests += 1
                else:
                    failed_tests += 1
                total_tests += 1

        print(f"\n📊 SUMMARY:")
        print(f"   Total Test Suites: {len(results)}")
        print(f"   Passed: {sum(1 for r in results.values() if r['success'])}")
        print(f"   Failed: {sum(1 for r in results.values() if not r['success'])}")

        if total_tests > 0:
            success_rate = (passed_tests / total_tests) * 100
            print(".1f"
            if success_rate >= 90:
                print("🎉 EXCELLENT - High test success rate!")
            elif success_rate >= 75:
                print("✅ GOOD - Acceptable test success rate")
            else:
                print("⚠️ NEEDS ATTENTION - Test success rate could be improved")

        return results

    def run_selected_tests(self, test_types):
        """Run selected test types."""
        results = {}

        if 'lint' in test_types:
            results.update(self.run_linting())

        if 'unit' in test_types:
            results['Unit Tests'] = self.run_unit_tests()

        if 'integration' in test_types:
            results['Integration Tests'] = self.run_integration_tests()

        if 'security' in test_types:
            results['Security Tests'] = self.run_security_tests()

        if 'performance' in test_types:
            results['Performance Tests'] = self.run_performance_tests()

        if 'all' in test_types:
            results['All Tests'] = self.run_all_tests()

        return self.generate_report(results)


def main():
    parser = argparse.ArgumentParser(description="Chatbot Test Runner")
    parser.add_argument(
        'tests',
        nargs='+',
        choices=['lint', 'unit', 'integration', 'security', 'performance', 'all'],
        help='Test types to run'
    )
    parser.add_argument(
        '--ci',
        action='store_true',
        help='Run in CI mode (exit with error code on failure)'
    )

    args = parser.parse_args()

    # Set up environment for testing
    os.environ['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY', 'test_key_placeholder')
    os.environ['GCHAT_WEBHOOK_URL'] = os.environ.get('GCHAT_WEBHOOK_URL', 'https://test-webhook.com')
    os.environ['GCP_PROJECT_ID'] = os.environ.get('GCP_PROJECT_ID', 'test-project')

    runner = TestRunner()
    results = runner.run_selected_tests(args.tests)

    # Check if any tests failed
    has_failures = any(not result['success'] for result in results.values())

    if args.ci and has_failures:
        print("\n❌ CI MODE: Tests failed, exiting with error code")
        sys.exit(1)
    elif has_failures:
        print("\n⚠️ Some tests failed, but continuing...")
        sys.exit(0)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
