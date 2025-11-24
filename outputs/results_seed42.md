# Run results — seed 42

- Date (UTC): 2025-11-24T16:26:35.715343Z

- GA-GAM Final Test RMSE: 0.7312

- Decision Tree Test RMSE: 0.6925

- Baseline PyGAM Test RMSE: 0.7216


## Generation Log

| Gen | Best Fitness |
|---:|---:|

| 1 | -0.753422 |

| 2 | -0.753523 |

| 3 | -0.753769 |

| 4 | -0.748009 |

| 5 | -0.746374 |


## Model Structure Summary

- **MedInc**: spline

- **HouseAge**: none

- **AveRooms**: linear

- **AveBedrms**: none

- **Population**: none

- **AveOccup**: none

- **Latitude**: spline

- **Longitude**: linear

- **MedInc_baseline**: spline

- **HouseAge_baseline**: spline

- **AveRooms_baseline**: spline

- **AveBedrms_baseline**: spline

- **Population_baseline**: spline

- **AveOccup_baseline**: spline

- **Latitude_baseline**: spline

- **Longitude_baseline**: spline


## Best Chromosome (JSON)

```
[
  {
    "type": "spline",
    "scale": true,
    "knots": 7,
    "lambda": 0.5498162913036825
  },
  {
    "type": "none",
    "scale": true,
    "knots": null,
    "lambda": null
  },
  {
    "type": "linear",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "none",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "none",
    "scale": true,
    "knots": null,
    "lambda": null
  },
  {
    "type": "none",
    "scale": true,
    "knots": null,
    "lambda": null
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 6,
    "lambda": 0.563494019856605
  },
  {
    "type": "linear",
    "scale": true,
    "knots": null,
    "lambda": null
  }
]
```