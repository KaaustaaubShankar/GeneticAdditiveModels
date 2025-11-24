# Run results — seed 7

- Date (UTC): 2025-11-24T16:28:38.636411Z

- GA-GAM Final Test RMSE: 0.6937

- Decision Tree Test RMSE: 0.7159

- Baseline PyGAM Test RMSE: 0.6674


## Generation Log

| Gen | Best Fitness |
|---:|---:|

| 1 | -0.761808 |

| 2 | -0.750910 |

| 3 | -0.731248 |

| 4 | -0.746142 |

| 5 | -0.743753 |


## Model Structure Summary

- **MedInc**: spline

- **HouseAge**: none

- **AveRooms**: none

- **AveBedrms**: linear

- **Population**: none

- **AveOccup**: spline

- **Latitude**: spline

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
    "lambda": 0.342053951504124
  },
  {
    "type": "none",
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
    "type": "spline",
    "scale": true,
    "knots": 9,
    "lambda": 0.7949078593629303
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 6,
    "lambda": 0.9298967294348329
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 8,
    "lambda": 0.9825318184696306
  }
]
```