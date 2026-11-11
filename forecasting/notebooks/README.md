# notebooks/

Exploration only. Notebooks may import from `forecasting.*` but must not define
logic that results depend on: anything reusable moves into the package and gets a
test. Typical start:

```python
from forecasting.cli import load_panel
from forecasting.config import load_config

cfg = load_config(data_root="/path/to/export")
panel, report = load_panel(cfg)
```
