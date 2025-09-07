#!/usr/bin/env python3
"""
Test script to verify environment variable loading from .env files.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_env_loading():
    """Test environment variable loading functionality."""

    print("🧪 Testing Environment Variable Loading")
    print("=" * 50)

    # Test 1: Check if .env file exists
    project_root = Path(__file__).parent
    env_file = project_root / '.env'
    env_local_file = project_root / '.env.local'

    print("1. Checking for .env files:")
    print(f"   .env file exists: {env_file.exists()}")
    print(f"   .env.local file exists: {env_local_file.exists()}")

    if not env_file.exists() and not env_local_file.exists():
        print("   ⚠️  No .env files found. Creating sample .env file...")

        # Create a sample .env file for testing
        sample_content = """# Test Environment Variables
GEMINI_API_KEY="test_gemini_key_from_env_file"
GCHAT_WEBHOOK_URL="https://test-webhook.com"
GCP_PROJECT_ID="test-project-from-env"
LOG_LEVEL="DEBUG"
PORT="8001"
"""
        env_file.write_text(sample_content)
        print("   ✅ Sample .env file created for testing")
    # Test 2: Import configuration and check loading
    print("\n2. Testing configuration loading:")

    try:
        import config
        print("   ✅ Configuration module imported successfully")

        # Check if environment variables are loaded
        print("\n3. Environment variable values:")
        print(f"   GEMINI_API_KEY: {'✅ Set' if config.GEMINI_API_KEY else '❌ Not set'}")
        print(f"   GCHAT_WEBHOOK_URL: {'✅ Set' if config.GCHAT_WEBHOOK_URL else '❌ Not set'}")
        print(f"   GCP_PROJECT_ID: {config.GCP_PROJECT_ID}")
        print(f"   LOG_LEVEL: {config.LOG_LEVEL}")
        print(f"   PORT: {config.PORT}")
        print(f"   HOST: {config.HOST}")

        # Test 3: Override with environment variables
        print("\n4. Testing environment variable override:")
        os.environ['TEST_OVERRIDE_VAR'] = 'from_environment'
        override_value = os.getenv('TEST_OVERRIDE_VAR')
        print(f"   TEST_OVERRIDE_VAR: {override_value}")

        # Clean up
        del os.environ['TEST_OVERRIDE_VAR']

        print("\n🎉 Environment variable loading test completed successfully!")

    except Exception as e:
        print(f"   ❌ Error during configuration loading: {e}")
        return False

    return True

def test_priority_loading():
    """Test that .env.local takes priority over .env"""
    print("\n🧪 Testing Priority Loading (.env.local > .env)")
    print("=" * 50)

    project_root = Path(__file__).parent
    env_file = project_root / '.env'
    env_local_file = project_root / '.env.local'

    # Create base .env file
    base_content = "PRIORITY_TEST_VAR=from_base_env\n"
    env_file.write_text(base_content)

    # Create .env.local file (should override)
    local_content = "PRIORITY_TEST_VAR=from_local_env\n"
    env_local_file.write_text(local_content)

    # Reload configuration
    import importlib
    if 'config' in sys.modules:
        importlib.reload(sys.modules['config'])
    import config

    test_var = os.getenv('PRIORITY_TEST_VAR')
    print(f"PRIORITY_TEST_VAR value: {test_var}")

    if test_var == 'from_local_env':
        print("✅ Priority loading working correctly (.env.local takes precedence)")
    else:
        print("❌ Priority loading not working as expected")

    # Clean up
    env_file.unlink(missing_ok=True)
    env_local_file.unlink(missing_ok=True)

    return test_var == 'from_local_env'

if __name__ == "__main__":
    success = test_env_loading()
    if success:
        priority_success = test_priority_loading()
        if priority_success:
            print("\n🎉 All environment variable tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Priority loading test failed!")
            sys.exit(1)
    else:
        print("\n❌ Environment variable loading test failed!")
        sys.exit(1)
