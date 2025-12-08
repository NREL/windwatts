import { ReactNode } from "react";

// Logo URLs
export const ALLIANCE_LOGO =
  "https://www.nrel.gov/images/libraries/global/alliance-logo-black.svg";
export const ALLIANCE_URL = "https://www.allianceforsustainableenergy.org/";
export const DOE_LOGO =
  "https://www.nrel.gov/images/libraries/about-images/u-s-department-of-energy.svg";
export const DOE_URL = "https://www.energy.gov/";

// Main navigation links
export interface FooterLink {
  content: string | ReactNode;
  href: string;
  "aria-label"?: string;
}

export const NAV_LINKS: FooterLink[] = [
  { content: "About", href: "https://www.nrel.gov/about" },
  { content: "Research", href: "https://www.nrel.gov/research" },
  { content: "Partner with Us", href: "https://www.nrel.gov/workingwithus" },
  { content: "News", href: "https://www.nrel.gov/news" },
  { content: "Careers", href: "https://www.nrel.gov/careers" },
];

// Quick links
export const QUICK_LINKS: FooterLink[] = [
  {
    content: "Accessibility",
    href: "https://www.nrel.gov/accessibility.html",
  },
  { content: "Disclaimer", href: "https://www.nrel.gov/disclaimer.html" },
  {
    content: "Security and Privacy",
    href: "https://www.nrel.gov/security.html",
  },
  {
    content: "Site Feedback",
    href: "https://www.nrel.gov/webmaster.html",
  },
  { content: "Developers", href: "https://developer.nrel.gov" },
  { content: "Employees", href: "https://thesource.nrel.gov" },
];

// Social media links (content will be icons, added in component)
export const SOCIAL_LINKS = [
  {
    name: "facebook",
    href: "https://www.facebook.com/nationalrenewableenergylab",
    ariaLabel: "Follow NLR on Facebook",
  },
  {
    name: "instagram",
    href: "https://www.instagram.com/nationalrenewableenergylab",
    ariaLabel: "Follow NLR on Instagram",
  },
  {
    name: "linkedin",
    href: "https://www.linkedin.com/company/national-renewable-energy-laboratory",
    ariaLabel: "Follow NLR on LinkedIn",
  },
  {
    name: "youtube",
    href: "https://www.youtube.com/user/NRELPR",
    ariaLabel: "Follow NLR on YouTube",
  },
  {
    name: "x",
    href: "https://twitter.com/nrel",
    ariaLabel: "Follow NLR on X",
  },
  {
    name: "threads",
    href: "https://threads.net/@nationalrenewableenergylab",
    ariaLabel: "Follow NLR on Threads",
  },
];

// Contact information
export const CONTACT_EMAIL = "windwatts@nrel.gov";
export const CONTACT_LINKS: FooterLink[] = [
  { content: "Contact Us", href: `mailto:${CONTACT_EMAIL}` },
  { content: "Visit", href: "https://www.nrel.gov/about/visiting-nrel" },
  {
    content: "Subscribe to NLR",
    href: "https://www.nrel.gov/news/subscribe",
  },
];

// Attribution text
export const ATTRIBUTION_TEXT = {
  labName: "National Laboratory of the Rockies",
  department: "U.S. Department of Energy",
  office: "Office of Critical Minerals and Energy Innovation",
  contract: "Contract No. DE-AC36-08GO28308",
};
