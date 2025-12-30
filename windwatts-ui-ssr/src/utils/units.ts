export function convertWindspeed(speed = 0, units = "mph") {
  const value = units === "mph" ? (speed * 2.2369).toFixed(1) : speed.toFixed(1);
  return `${value} ${units === "mph" ? "mph" : "m/s"}`;
}

export function convertOutput(output = 0, units = "kWh") {
  const value = units === "MWh" ? output / 1000 : output;
  const rounded = units === "MWh" ? Number(value).toFixed(1) : Math.round(value).toString();
  return `${Number(rounded).toLocaleString("en-US")} ${units}`;
}
