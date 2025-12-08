import { Box, Stack, Link, Divider, Typography, SvgIcon } from "@mui/material";
import { Facebook, Instagram, LinkedIn, YouTube } from "@mui/icons-material";
import {
  ALLIANCE_LOGO,
  ALLIANCE_URL,
  DOE_LOGO,
  DOE_URL,
  NAV_LINKS,
  QUICK_LINKS,
  SOCIAL_LINKS,
  CONTACT_LINKS,
  ATTRIBUTION_TEXT,
} from "../../constants/footer";
// eslint-disable-next-line
import XIconSvg from "./XIcon.svg?react";
// eslint-disable-next-line
import ThreadsIconSvg from "./ThreadsIcon.svg?react";

const XIcon = () => (
  <SvgIcon fontSize="inherit">
    <XIconSvg />
  </SvgIcon>
);

const ThreadsIcon = () => (
  <SvgIcon fontSize="inherit">
    <ThreadsIconSvg />
  </SvgIcon>
);

const getSocialIcon = (name: string) => {
  switch (name) {
    case "facebook":
      return <Facebook fontSize="inherit" />;
    case "instagram":
      return <Instagram fontSize="inherit" />;
    case "linkedin":
      return <LinkedIn fontSize="inherit" />;
    case "youtube":
      return <YouTube fontSize="inherit" />;
    case "x":
      return <XIcon />;
    case "threads":
      return <ThreadsIcon />;
    default:
      return null;
  }
};

export const Footer = () => (
  <Box component="footer" color="black">
    <Box
      id="footer-top"
      sx={{
        background: "#D1D5D8",
        paddingBlock: "20px",
      }}
    >
      <Box
        sx={{
          paddingInline: "30px",
          maxWidth: { xs: 540, md: 720, lg: 960 },
          fontSize: "0.9rem",
        }}
      >
        <Stack direction="column" spacing="10px">
          <Box>
            <Link href="https://www.nrel.gov" underline="none" color="inherit">
              <strong>{ATTRIBUTION_TEXT.labName}</strong>
            </Link>
          </Box>
          <Stack component="ul" aria-label="footer navigation">
            {NAV_LINKS.map((link) => (
              <li
                key={link.href}
                style={{ listStyleType: "none", fontSize: "0.8rem" }}
              >
                <Link
                  href={link.href}
                  aria-label={link["aria-label"]}
                  underline="none"
                  color="inherit"
                >
                  {link.content}
                </Link>
              </li>
            ))}
          </Stack>
        </Stack>
      </Box>
    </Box>

    <Box
      id="footer-bottom"
      sx={{
        color: "black",
        background: "#F5F5F5",
        padding: "30px",
      }}
    >
      <Box sx={{ maxWidth: { xs: 540, md: 720, lg: 960 } }}>
        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing="20px"
          sx={{ marginBottom: "20px" }}
        >
          <Stack sx={{ flex: { md: "40%" } }}>
            {CONTACT_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                fontSize="0.8rem"
                color="inherit"
                underline="hover"
                sx={{ display: "block" }}
              >
                {link.content}
              </Link>
            ))}
            <Stack
              component="ul"
              aria-label="social links"
              direction="row"
              spacing={1}
              marginTop={2}
            >
              {SOCIAL_LINKS.map((link) => (
                <li key={link.href} style={{ listStyleType: "none" }}>
                  <Link
                    href={link.href}
                    aria-label={link.ariaLabel}
                    color="inherit"
                  >
                    {getSocialIcon(link.name)}
                  </Link>
                </li>
              ))}
            </Stack>
          </Stack>

          <Stack
            component="ul"
            aria-label="footer quick links"
            sx={{
              flex: { md: "60%" },
              flexWrap: "wrap",
              maxHeight: { md: "80px" },
              fontSize: "0.7rem",
            }}
          >
            {QUICK_LINKS.map((link) => (
              <li key={link.href} style={{ listStyleType: "none" }}>
                <Link
                  href={link.href}
                  aria-label={link["aria-label"]}
                  color="inherit"
                  underline="hover"
                >
                  {link.content}
                </Link>
              </li>
            ))}
          </Stack>
        </Stack>

        <Divider color="#D1D5D8" sx={{ marginBlock: "1rem" }} />

        <Stack spacing="1rem">
          <Stack
            direction={{ xs: "column", md: "row" }}
            spacing="2rem"
            sx={{ alignItems: { xs: "flex-start", md: "center" } }}
          >
            <a href={ALLIANCE_URL}>
              <img
                src={ALLIANCE_LOGO}
                height={70}
                alt="Alliance for Sustainable Energy, LLC"
              />
            </a>
            <a href={DOE_URL}>
              <img src={DOE_LOGO} height={70} alt="U.S. Department of Energy" />
            </a>
          </Stack>
          <Typography variant="caption" sx={{ lineHeight: "1rem" }}>
            {`The ${ATTRIBUTION_TEXT.labName} is a national laboratory of the ${ATTRIBUTION_TEXT.department}, ${ATTRIBUTION_TEXT.office}, operated under ${ATTRIBUTION_TEXT.contract}.`}
          </Typography>
        </Stack>
      </Box>
    </Box>
  </Box>
);
