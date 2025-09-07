#!/usr/bin/env python3
"""
Comprehensive project analysis and testing script for the multi-agent chatbot.
"""

import os
import sys
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Any
import ast
import re

class ProjectAnalyzer:
    """Comprehensive analyzer for the multi-agent chatbot project."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues = []
        self.warnings = []
        self.recommendations = []
        self.analysis_results = {}

    def run_full_analysis(self):
        """Run complete project analysis."""
        print("🔍 Starting comprehensive project analysis...")
        print("=" * 60)

        # Structural analysis
        self.analyze_project_structure()
        self.analyze_code_quality()
        self.analyze_configuration()
        self.analyze_dependencies()
        self.analyze_security()

        # Feature analysis
        self.analyze_multi_agent_system()
        self.analyze_frontend_backend_integration()
        self.analyze_error_handling()
        self.analyze_performance()

        # Generate report
        self.generate_analysis_report()

        return self.analysis_results

    def analyze_project_structure(self):
        """Analyze project structure and organization."""
        print("\n📁 Analyzing project structure...")

        required_dirs = [
            "app", "app/agents", "app/services", "static", "utils", "data"
        ]

        required_files = [
            "app/main.py", "requirements.txt", "README.md", "config.py"
        ]

        # Check directories
        for dir_path in required_dirs:
            if not (self.project_root / dir_path).exists():
                self.issues.append(f"Missing required directory: {dir_path}")
            else:
                print(f"✓ Found directory: {dir_path}")

        # Check files
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                self.issues.append(f"Missing required file: {file_path}")
            else:
                print(f"✓ Found file: {file_path}")

        # Analyze agent files
        agent_files = [
            "app/agents/__init__.py",
            "app/agents/orchestrator.py",
            "app/agents/professional_agent.py",
            "app/agents/education_agent.py",
            "app/agents/learning_agent.py",
            "app/agents/redirect_agent.py",
            "app/agents/retrievers.py"
        ]

        for agent_file in agent_files:
            if (self.project_root / agent_file).exists():
                print(f"✓ Found agent component: {agent_file}")
            else:
                self.warnings.append(f"Missing agent component: {agent_file}")

        # Analyze service files
        service_files = [
            "app/services/__init__.py",
            "app/services/dynamic_guardrails.py",
            "app/services/google_chat_alert.py",
            "app/services/language_detection.py"
        ]

        for service_file in service_files:
            if (self.project_root / service_file).exists():
                print(f"✓ Found service component: {service_file}")
            else:
                self.warnings.append(f"Missing service component: {service_file}")

    def analyze_code_quality(self):
        """Analyze code quality and patterns."""
        print("\n💻 Analyzing code quality...")

        python_files = list(self.project_root.rglob("*.py"))
        total_lines = 0
        total_functions = 0
        total_classes = 0

        for py_file in python_files:
            if py_file.name.startswith('.') or 'pycache' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.split('\n'))
                    total_lines += lines

                # Parse AST for code analysis
                try:
                    tree = ast.parse(content)
                    functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
                    classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])

                    total_functions += functions
                    total_classes += classes

                    print(f"  {py_file.name}: {lines} lines, {functions} functions, {classes} classes")

                except SyntaxError as e:
                    self.issues.append(f"Syntax error in {py_file}: {e}")

            except Exception as e:
                self.warnings.append(f"Could not analyze {py_file}: {e}")

        print(f"\n📊 Code Statistics:")
        print(f"  Total Python files: {len([f for f in python_files if not f.name.startswith('.') and 'pycache' not in str(f)])}")
        print(f"  Total lines of code: {total_lines}")
        print(f"  Total functions: {total_functions}")
        print(f"  Total classes: {total_classes}")

        # Code quality checks
        self.check_code_quality(python_files)

    def check_code_quality(self, python_files):
        """Check for common code quality issues."""
        print("\n🔍 Code Quality Checks:")

        issues_found = 0

        for py_file in python_files:
            if py_file.name.startswith('.') or 'pycache' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                # Check for TODO comments
                todo_count = sum(1 for line in lines if 'TODO' in line.upper())
                if todo_count > 0:
                    self.warnings.append(f"{py_file.name}: {todo_count} TODO comments found")

                # Check for print statements (should use logging)
                print_count = sum(1 for line in lines if line.strip().startswith('print('))
                if print_count > 10:  # Allow some prints
                    self.warnings.append(f"{py_file.name}: {print_count} print statements (consider using logging)")

                # Check for long functions
                if len(lines) > 300:
                    self.warnings.append(f"{py_file.name}: Very long file ({len(lines)} lines)")

                # Check for missing docstrings
                if 'def ' in content and '"""' not in content:
                    self.warnings.append(f"{py_file.name}: Missing docstrings")

            except Exception as e:
                self.warnings.append(f"Could not check quality for {py_file}: {e}")

        if issues_found == 0:
            print("✓ No major code quality issues found")

    def analyze_configuration(self):
        """Analyze configuration setup."""
        print("\n⚙️ Analyzing configuration...")

        config_file = self.project_root / "config.py"
        if config_file.exists():
            print("✓ Configuration file found")

            with open(config_file, 'r', encoding='utf-8') as f:
                config_content = f.read()

            # Check for environment variables
            env_vars = re.findall(r'os\.getenv\("([^"]+)"', config_content)
            print(f"  Environment variables used: {len(env_vars)}")
            for var in env_vars:
                print(f"    - {var}")

            # Check for validation
            if 'validate_config' in config_content:
                print("✓ Configuration validation found")
            else:
                self.warnings.append("No configuration validation found")

        else:
            self.issues.append("Configuration file not found")

    def analyze_dependencies(self):
        """Analyze project dependencies."""
        print("\n📦 Analyzing dependencies...")

        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            print("✓ Requirements file found")

            with open(req_file, 'r', encoding='utf-8') as f:
                deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]

            print(f"  Dependencies: {len(deps)}")
            for dep in deps:
                print(f"    - {dep}")

            # Check for common ML/AI libraries
            ml_libs = ['langchain', 'google-generativeai', 'chromadb', 'fastapi']
            found_ml = [lib for lib in ml_libs if any(lib in dep for dep in deps)]

            if len(found_ml) >= 3:
                print("✓ Core ML/AI dependencies found")
            else:
                self.warnings.append("Some core ML/AI dependencies might be missing")

        else:
            self.issues.append("Requirements file not found")

    def analyze_security(self):
        """Analyze security aspects."""
        print("\n🔒 Analyzing security...")

        # Check for sensitive files
        sensitive_patterns = [
            ".env", ".key", "secret", "password", "token", "credential"
        ]

        for pattern in sensitive_patterns:
            sensitive_files = list(self.project_root.rglob(f"*{pattern}*"))
            if sensitive_files:
                for file in sensitive_files:
                    if not file.name.startswith('.') or 'test' in file.name.lower():
                        self.warnings.append(f"Potential sensitive file found: {file}")

        # Check for hardcoded secrets in code
        python_files = list(self.project_root.rglob("*.py"))
        for py_file in python_files:
            if 'pycache' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for potential hardcoded secrets
                secret_patterns = [
                    r'password\s*=\s*["\'][^"\']+["\']',
                    r'secret\s*=\s*["\'][^"\']+["\']',
                    r'key\s*=\s*["\'][^"\']+["\']',
                    r'token\s*=\s*["\'][^"\']+["\']'
                ]

                for pattern in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        self.warnings.append(f"Potential hardcoded secret in {py_file.name}")

            except Exception as e:
                self.warnings.append(f"Could not analyze security for {py_file}: {e}")

        print("✓ Security analysis completed")

    def analyze_multi_agent_system(self):
        """Analyze multi-agent system implementation."""
        print("\n🤖 Analyzing multi-agent system...")

        # Check agent files
        agent_components = {
            "orchestrator": "app/agents/orchestrator.py",
            "professional_agent": "app/agents/professional_agent.py",
            "education_agent": "app/agents/education_agent.py",
            "learning_agent": "app/agents/learning_agent.py",
            "redirect_agent": "app/agents/redirect_agent.py"
        }

        agent_score = 0
        for name, path in agent_components.items():
            if (self.project_root / path).exists():
                agent_score += 1
                print(f"✓ {name} component found")
            else:
                self.issues.append(f"Missing {name} component")

        # Check for advanced features
        advanced_features = [
            "session_management",
            "dynamic_guardrails",
            "language_detection",
            "google_chat_integration"
        ]

        for feature in advanced_features:
            feature_path = f"app/services/{feature.replace('_', '')}.py"
            if (self.project_root / feature_path).exists():
                print(f"✓ Advanced feature: {feature}")
                agent_score += 1
            else:
                self.warnings.append(f"Missing advanced feature: {feature}")

        print(f"\n📊 Multi-Agent Score: {agent_score}/9 components")

        if agent_score >= 7:
            print("🎉 Multi-agent system is well implemented!")
        elif agent_score >= 5:
            print("✅ Multi-agent system is partially implemented")
        else:
            print("⚠️ Multi-agent system needs improvement")

    def analyze_frontend_backend_integration(self):
        """Analyze frontend-backend integration."""
        print("\n🌐 Analyzing frontend-backend integration...")

        # Check API endpoints in main.py
        main_file = self.project_root / "app/main.py"
        if main_file.exists():
            with open(main_file, 'r', encoding='utf-8') as f:
                main_content = f.read()

            # Count API endpoints
            endpoint_patterns = [
                r'@app\.(get|post|put|delete)\(',
                r'@app\.websocket\('
            ]

            endpoint_count = 0
            for pattern in endpoint_patterns:
                endpoint_count += len(re.findall(pattern, main_content))

            print(f"✓ Found {endpoint_count} API endpoints")

            # Check for CORS, error handling, etc.
            if 'CORSMiddleware' in main_content:
                print("✓ CORS middleware configured")
            else:
                self.warnings.append("CORS middleware not found")

        # Check frontend files
        frontend_files = [
            "static/index.html",
            "static/script.js",
            "static/style.css"
        ]

        frontend_score = 0
        for file in frontend_files:
            if (self.project_root / file).exists():
                frontend_score += 1
                print(f"✓ Frontend file: {file}")
            else:
                self.warnings.append(f"Missing frontend file: {file}")

        print(f"📊 Frontend Score: {frontend_score}/3 files")

    def analyze_error_handling(self):
        """Analyze error handling patterns."""
        print("\n🚨 Analyzing error handling...")

        python_files = list(self.project_root.rglob("*.py"))
        error_patterns = [
            r'try:',
            r'except\s+',
            r'finally:',
            r'raise\s+',
            r'logging\.',
            r'logger\.'
        ]

        error_handling_score = 0
        for py_file in python_files:
            if 'pycache' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                file_score = 0
                for pattern in error_patterns:
                    if re.search(pattern, content):
                        file_score += 1

                if file_score > 0:
                    error_handling_score += 1
                    print(f"✓ Error handling in {py_file.name}")

            except Exception as e:
                self.warnings.append(f"Could not analyze error handling for {py_file}: {e}")

        print(f"📊 Error Handling Score: {error_handling_score}/{len([f for f in python_files if 'pycache' not in str(f)])} files")

    def analyze_performance(self):
        """Analyze performance considerations."""
        print("\n⚡ Analyzing performance...")

        # Check for async/await patterns
        python_files = list(self.project_root.rglob("*.py"))
        async_count = 0

        for py_file in python_files:
            if 'pycache' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                if 'async def' in content or 'await ' in content:
                    async_count += 1
                    print(f"✓ Async patterns in {py_file.name}")

            except Exception as e:
                self.warnings.append(f"Could not analyze performance for {py_file}: {e}")

        print(f"📊 Async/Await Usage: {async_count} files")

        # Check for caching, optimization patterns
        optimization_patterns = [
            'cache', 'memoize', 'lru_cache', 'functools.cache'
        ]

        for py_file in python_files:
            if 'pycache' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern in optimization_patterns:
                    if pattern in content:
                        print(f"✓ Optimization pattern '{pattern}' found in {py_file.name}")
                        break

            except Exception as e:
                pass

    def generate_analysis_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "=" * 60)
        print("📋 PROJECT ANALYSIS REPORT")
        print("=" * 60)

        # Issues
        if self.issues:
            print(f"\n🚨 ISSUES FOUND ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\n✅ No critical issues found!")

        # Warnings
        if self.warnings:
            print(f"\n⚠️ WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        # Overall assessment
        issue_score = len(self.issues)
        warning_score = len(self.warnings)

        if issue_score == 0 and warning_score == 0:
            print("\n🎉 PROJECT STATUS: EXCELLENT")
            print("   All components are properly implemented!")
        elif issue_score == 0 and warning_score <= 5:
            print("\n✅ PROJECT STATUS: GOOD")
            print("   Minor improvements needed")
        elif issue_score <= 3:
            print("\n⚠️ PROJECT STATUS: NEEDS ATTENTION")
            print("   Some issues require fixing")
        else:
            print("\n🚨 PROJECT STATUS: NEEDS IMPROVEMENT")
            print("   Multiple issues need to be addressed")

        # Generate recommendations
        self.generate_recommendations()

        return {
            "issues": self.issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "status": "excellent" if issue_score == 0 and warning_score == 0 else
                     "good" if issue_score == 0 and warning_score <= 5 else
                     "needs_attention" if issue_score <= 3 else "needs_improvement"
        }

    def generate_recommendations(self):
        """Generate improvement recommendations."""
        print("\n💡 RECOMMENDATIONS:")

        recommendations = []

        # Based on analysis results
        if not any("agent" in str(f) for f in self.project_root.rglob("*.py")):
            recommendations.append("Implement comprehensive test coverage for all agents")

        if len(list(self.project_root.rglob("*.py"))) > 20:
            recommendations.append("Consider breaking down large files into smaller modules")

        if not any("async" in open(f).read() for f in self.project_root.rglob("*.py") if "pycache" not in str(f)):
            recommendations.append("Implement async/await patterns for better performance")

        recommendations.extend([
            "Add comprehensive error handling and logging",
            "Implement rate limiting for API endpoints",
            "Add monitoring and metrics collection",
            "Create automated deployment pipeline",
            "Add comprehensive API documentation",
            "Implement user feedback collection system",
            "Add performance monitoring and optimization",
            "Create backup and recovery procedures",
            "Implement A/B testing capabilities",
            "Add analytics and usage tracking"
        ])

        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

        self.recommendations = recommendations

def main():
    """Main analysis function."""
    project_root = "/Users/bolablg/Desktop/BOLABLG.com/deployed_projects/chatbot_portfolio/agentic-rag-chatbot"

    analyzer = ProjectAnalyzer(project_root)
    results = analyzer.run_full_analysis()

    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS TO IMPROVE THE PROJECT")
    print("=" * 60)

    print("\n🚀 IMMEDIATE ACTIONS (Priority 1):")
    print("  1. Fix configuration validation to allow optional environment variables")
    print("  2. Add comprehensive error handling throughout the application")
    print("  3. Implement proper logging configuration")
    print("  4. Add input validation for all API endpoints")
    print("  5. Create unit tests for all agent components")

    print("\n📈 SHORT-TERM IMPROVEMENTS (Priority 2):")
    print("  1. Add async/await patterns for better concurrency")
    print("  2. Implement caching mechanisms for frequently accessed data")
    print("  3. Add rate limiting and request throttling")
    print("  4. Create comprehensive API documentation with OpenAPI/Swagger")
    print("  5. Add monitoring and metrics collection")

    print("\n🔬 MEDIUM-TERM ENHANCEMENTS (Priority 3):")
    print("  1. Implement A/B testing framework for agent responses")
    print("  2. Add user feedback collection and analysis")
    print("  3. Create automated deployment and rollback procedures")
    print("  4. Implement advanced analytics and usage tracking")
    print("  5. Add support for additional languages and locales")

    print("\n🚀 LONG-TERM VISION (Priority 4):")
    print("  1. Implement machine learning for response optimization")
    print("  2. Add voice interaction capabilities")
    print("  3. Create mobile applications for iOS and Android")
    print("  4. Implement advanced conversation flows and state management")
    print("  5. Add integration with additional platforms and services")

    print("\n💡 DEVELOPMENT WORKFLOW IMPROVEMENTS:")
    print("  1. Set up automated testing pipeline")
    print("  2. Implement code quality checks and linting")
    print("  3. Add pre-commit hooks for code standards")
    print("  4. Create development environment setup scripts")
    print("  5. Implement feature flags for gradual rollouts")

    return results

if __name__ == "__main__":
    main()
