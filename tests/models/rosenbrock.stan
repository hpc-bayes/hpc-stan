parameters {
  real x;
  real y;
}
model {
  target += -((1 - x) ^ 2 + 100 * (y - x ^ 2) ^ 2);
}
