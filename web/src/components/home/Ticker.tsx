/** A slow band of repeating facts, like the reference's marquee. It stands still under reduced
 *  motion (the global CSS rule stops the animation). */
export function Ticker({ items }: { items: string[] }) {
  const run = [...items, ...items, ...items];
  return (
    <div className="overflow-hidden border-b border-line bg-sand py-4" aria-label={items.join(", ")}>
      <div className="ticker flex w-max gap-10 whitespace-nowrap font-mono text-xs uppercase tracking-[0.22em] text-muted" aria-hidden>
        {[...run, ...run].map((item, i) => (
          <span key={i} className="flex items-center gap-10">
            {item}
            <span className="text-sand-deep">✦</span>
          </span>
        ))}
      </div>
    </div>
  );
}
