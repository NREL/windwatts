export const getWindResource = (speed: number) => {
  return speed > 5 ? "High" : speed >= 3 ? "Moderate" : "Low";
};
