# Run results — seed 123

- Date (UTC): 2025-11-24T16:30:40.775385Z

- GA-GAM Final Test RMSE: 0.7530

- Decision Tree Test RMSE: 0.6961

- Baseline PyGAM Test RMSE: 0.6697


## Generation Log

| Gen | Best Fitness |
|---:|---:|

| 1 | -0.749432 |

| 2 | -0.738794 |

| 3 | -0.736194 |

| 4 | -0.733016 |

| 5 | -0.720546 |


## Model Structure Summary

- **MedInc**: spline

- **HouseAge**: spline

- **AveRooms**: linear

- **AveBedrms**: none

- **Population**: linear

- **AveOccup**: none

- **Latitude**: none

- **Longitude**: spline

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
    "knots": 9,
    "lambda": 0.8392078723026983
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 9,
    "lambda": 0.28438851462724385
  },
  {
    "type": "linear",
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
    "type": "linear",
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
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 6,
    "lambda": 0.260513213737766
  }
]
```