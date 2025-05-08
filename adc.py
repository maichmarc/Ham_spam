import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent # Adjust based on your structure
# print(project_root)
print(sys.path)
# sys.path.append(str(project_root))