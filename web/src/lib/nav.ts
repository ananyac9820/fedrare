/** The order of the site, start to end. Used by the header, the pager and the flow page. */
export const NAV = [
  { href: "/", label: "Home", blurb: "What this project is" },
  { href: "/flow", label: "Flow", blurb: "The whole system, end to end" },
  { href: "/problem", label: "Problem", blurb: "Knowledge does not match weight" },
  { href: "/method", label: "Method", blurb: "What EARN would do" },
  { href: "/results", label: "Results", blurb: "The gates, including the failures" },
  { href: "/study", label: "Study", blurb: "408 runs: methods under attack" },
  { href: "/ledger", label: "Ledger", blurb: "The tamper-proof record" },
  { href: "/status", label: "Status", blurb: "Where it stands, what is left" },
] as const;

export type NavItem = (typeof NAV)[number];

export const navIndex = (pathname: string) =>
  NAV.findIndex((n) => (n.href === "/" ? pathname === "/" : pathname.startsWith(n.href)));

export const isActive = (pathname: string, href: string) =>
  href === "/" ? pathname === "/" : pathname.startsWith(href);
