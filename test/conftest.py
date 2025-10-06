import sys
from pathlib import Path
from unittest.mock import Mock

# Add project directories to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Mock any external dependencies that aren't needed for unit tests
# This prevents import errors from external packages
sys.modules['dotenv'] = Mock()