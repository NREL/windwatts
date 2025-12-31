import { Box, Link, Typography } from "@mui/material";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";

export function OutOfBoundsWarning({ message }: { message: string }) {
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "flex-start",
        gap: 1,
        bgcolor: "#fff3cd",
        color: "#856404",
        border: "1px solid #ffeeba",
        borderRadius: 1,
        p: 1.5,
        fontSize: 14,
        minHeight: 60,
      }}
    >
      <WarningAmberIcon sx={{ color: "#ff9800", fontSize: 22, mt: "2px" }} />
      <Box>
        <Typography
          variant="subtitle2"
          sx={{ fontWeight: 600, mb: 0.25, fontSize: 15 }}
        >
          Out of Bounds
        </Typography>
        <Typography
          variant="body2"
          sx={{ whiteSpace: "pre-line", fontSize: 13 }}
        >
          {message}
        </Typography>
        <Typography variant="body2" sx={{ mt: 1, fontSize: 12 }}>
          <Link
            href="https://github.com/NREL/windwatts/blob/main/docs/about/era5.md#spatial-coverage"
            target="_blank"
            rel="noopener noreferrer"
            underline="hover"
            sx={{ color: "#856404", fontWeight: 500 }}
          >
            Learn more about spatial coverage
          </Link>
        </Typography>
      </Box>
    </Box>
  );
}
