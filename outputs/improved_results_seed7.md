# Improved Run results — seed 7

- Date (UTC): 2025-11-25T19:37:48.074117Z

- Improved GA-GAM Final Test RMSE: 0.6761

- Decision Tree Test RMSE: 0.7159

- Baseline PyGAM Test RMSE: 0.6674


## Generation Log

| Gen | Best Fitness | Average Fitness |
|---:|---:|---:|

| 0 | -0.690763 | -inf |

| 1 | -0.687085 | -inf |

| 2 | -0.677065 | -inf |

| 3 | -0.677065 | -inf |

| 4 | -0.673528 | -inf |

| 5 | -0.666175 | -inf |

| 6 | -0.666175 | -inf |

| 7 | -0.666175 | -inf |

| 8 | -0.664562 | -inf |

| 9 | -0.664562 | -inf |

| 10 | -0.664547 | -inf |

| 11 | -0.663006 | -inf |

| 12 | -0.659834 | -inf |

| 13 | -0.659272 | -inf |

| 14 | -0.658573 | -inf |

| 15 | -0.657101 | -inf |

| 16 | -0.656338 | -inf |

| 17 | -0.656176 | -inf |

| 18 | -0.655498 | -inf |

| 19 | -0.655412 | -inf |

| 20 | -0.655412 | -inf |

| 21 | -0.655408 | -inf |

| 22 | -0.655192 | -inf |

| 23 | -0.655142 | -inf |

| 24 | -0.655142 | -inf |

| 25 | -0.655142 | -inf |

| 26 | -0.655139 | -inf |

| 27 | -0.655058 | -inf |

| 28 | -0.655058 | -inf |

| 29 | -0.655029 | -inf |

| 30 | -0.654943 | -inf |

| 31 | -0.654907 | -inf |

| 32 | -0.654907 | -inf |

| 33 | -0.654907 | -inf |

| 34 | -0.654907 | -inf |

| 35 | -0.654907 | -inf |

| 36 | -0.654904 | -inf |

| 37 | -0.654743 | -inf |

| 38 | -0.654743 | -inf |

| 39 | -0.654743 | -inf |

| 40 | -0.654739 | -inf |

| 41 | -0.654739 | -inf |

| 42 | -0.654739 | -inf |

| 43 | -0.654495 | -inf |

| 44 | -0.654492 | -inf |

| 45 | -0.654492 | -inf |

| 46 | -0.654393 | -inf |

| 47 | -0.654391 | -inf |

| 48 | -0.654391 | -inf |

| 49 | -0.654391 | -inf |

| 50 | -0.654388 | -inf |


## Model Structure Summary

- **MedInc**: spline

- **HouseAge**: spline

- **AveRooms**: spline

- **AveBedrms**: spline

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

```json
[
  {
    "type": "spline",
    "scale": true,
    "knots": 9,
    "lambda": 8.267409055174769
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 8,
    "lambda": 4.044812906535987
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 20,
    "lambda": 0.9944962268218194
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 15,
    "lambda": 0.25603580044899665
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
    "knots": 12,
    "lambda": 7.613477482756874
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 18,
    "lambda": 3.6426630613116258
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 15,
    "lambda": 3.52883952014102
  }
]
```