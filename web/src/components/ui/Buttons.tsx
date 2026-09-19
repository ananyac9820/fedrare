import Link from "next/link";

/** Rounded pill button-link, as in the reference. */
export function PillLink({ href, children, variant = "solid", external = false }: {
  href: string;
  children: React.ReactNode;
  variant?: "solid" | "outline";
  external?: boolean;
}) {
  const cls =
    variant === "solid"
      ? "bg-ink text-cream hover:bg-accent"
      : "border border-ink/20 bg-paper text-ink hover:border-ink/50";
  const className = `inline-flex items-center gap-2 rounded-full px-6 py-3 text-sm font-semibold transition ${cls}`;
  return external ? (
    <a href={href} className={className} target="_blank" rel="noreferrer">
      {children}
    </a>
  ) : (
    <Link href={href} className={className}>
      {children}
    </Link>
  );
}

/** A card - generous padding, soft border, paper surface. */
export function Card({ children, className = "", dashed = false }: {
  children: React.ReactNode;
  className?: string;
  dashed?: boolean;
}) {
  return (
    <div
      className={`rounded-[28px] border bg-paper p-8 md:p-10 ${dashed ? "border-dashed border-notrun/40" : "border-line"} ${className}`}
    >
      {children}
    </div>
  );
}
