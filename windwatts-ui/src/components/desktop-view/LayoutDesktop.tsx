import { Outlet, Link as RouterLink } from "react-router-dom";
import { Link, Box, AppBar, Toolbar } from "@mui/material";
import { Settings } from "../settings";
import { RightPane } from "../resultPane";
import { Footer } from "../Footer";

export function LayoutDesktop() {
  const APP_TITLE = import.meta.env.VITE_APP_TITLE || "Wind Watts";

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      <Box sx={{ flexGrow: 0 }}>
        <AppBar position="static" sx={{ bgcolor: "#0279c2" }}>
          <Toolbar sx={{ display: "flex", justifyContent: "space-between" }}>
            <Link
              to="/"
              variant="h5"
              component={RouterLink}
              underline="none"
              sx={{ flexGrow: 1, color: "white" }}
            >
              {APP_TITLE}
            </Link>
            <Box
              component="img"
              sx={{ height: 40 }}
              src="/assets/NLR-Blue-TypeTreatment-Block-Top-RGB.svg"
              alt="NLR Logo"
            />
          </Toolbar>
        </AppBar>
      </Box>

      <Box
        sx={{
          display: "flex",
          flexGrow: 1,
          overflow: "hidden",
        }}
      >
        <Box
          sx={{
            position: "relative",
            flexGrow: 1,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <Outlet />
        </Box>

        <Box
          className="right-pane"
          sx={{
            width: 420,
            bgcolor: "white",
            overflowY: "auto",
            borderLeft: "1px solid #ddd",
          }}
        >
          <Box>
            <RightPane />
          </Box>
          <Footer />
        </Box>
      </Box>
      <Settings />
    </Box>
  );
}
