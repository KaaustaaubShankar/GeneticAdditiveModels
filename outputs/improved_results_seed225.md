# Improved Run results — seed 225

- Date (UTC): 2025-11-25T20:28:42.493946Z

- Improved GA-GAM Final Test RMSE: 0.6411

- Decision Tree Test RMSE: 0.6770

- Baseline PyGAM Test RMSE: 0.6536


## Generation Log

| Gen | Best Fitness | Average Fitness |
|---:|---:|---:|

| 0 | -0.703757 | -inf |

| 1 | -0.702006 | -inf |

| 2 | -0.697783 | -inf |

| 3 | -0.687848 | -inf |

| 4 | -0.686428 | -inf |

| 5 | -0.685720 | -0.695458 |

| 6 | -0.683883 | -inf |

| 7 | -0.683646 | -inf |

| 8 | -0.682942 | -inf |

| 9 | -0.682007 | -inf |

| 10 | -0.681724 | -inf |

| 11 | -0.681608 | -inf |

| 12 | -0.681557 | -inf |

| 13 | -0.681498 | -inf |

| 14 | -0.681381 | -inf |

| 15 | -0.681378 | -inf |

| 16 | -0.677174 | -inf |

| 17 | -0.675915 | -inf |

| 18 | -0.675915 | -inf |

| 19 | -0.675906 | -inf |

| 20 | -0.675906 | -0.678042 |

| 21 | -0.675819 | -inf |

| 22 | -0.675794 | -inf |

| 23 | -0.675747 | -inf |

| 24 | -0.675725 | -inf |

| 25 | -0.675638 | -inf |

| 26 | -0.675635 | -inf |

| 27 | -0.675536 | -inf |

| 28 | -0.675488 | -inf |

| 29 | -0.675488 | -inf |

| 30 | -0.675488 | -inf |

| 31 | -0.675304 | -inf |

| 32 | -0.675304 | -inf |

| 33 | -0.675304 | -inf |

| 34 | -0.675304 | -inf |

| 35 | -0.675130 | -inf |

| 36 | -0.675130 | -inf |

| 37 | -0.675130 | -inf |

| 38 | -0.675125 | -inf |

| 39 | -0.675116 | -inf |

| 40 | -0.675116 | -inf |

| 41 | -0.675116 | -inf |

| 42 | -0.675116 | -inf |

| 43 | -0.675116 | -inf |

| 44 | -0.675116 | -inf |

| 45 | -0.675116 | -inf |

| 46 | -0.674792 | -inf |

| 47 | -0.674792 | -inf |

| 48 | -0.674792 | -inf |

| 49 | -0.674635 | -inf |

| 50 | -0.674586 | -inf |


## Model Structure Summary

- **MedInc**: spline

- **HouseAge**: spline

- **AveRooms**: spline

- **AveBedrms**: spline

- **Population**: none

- **AveOccup**: none

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
    "knots": 12,
    "lambda": 10.294820972438377
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 7,
    "lambda": 8.9638206295448
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 19,
    "lambda": 2.1361379752948184
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 20,
    "lambda": 0.2564939263041611
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
    "scale": false,
    "knots": 18,
    "lambda": 0.3829139216252535
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 20,
    "lambda": 1.9953162363187276
  }
]
```