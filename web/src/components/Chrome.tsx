import { StatusLegend, StatusTag, SampleBadge } from "@/components/ui/Status";

const LINKS = [
  ["#problem", "Problem"],
  ["#how", "How it works"],
  ["#results", "Results"],
  ["#ledger", "Ledger"],
  ["#status", "Status"],
] as const;

export function Nav() {
  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-line bg-ink-950/70 backdrop-blur-xl">
      <nav className="mx-auto flex h-14 max-w-7xl items-center gap-6 px-6">
        <a href="#top" className="font-display text-lg font-semibold tracking-tight text-white">
          EARN<span className="text-accent">.</span>
        </a>
        <ul className="hidden items-center gap-5 text-sm text-slate-400 md:flex">
          {LINKS.map(([href, label]) => (
            <li key={href}>
              <a href={href} className="transition hover:text-white">
                {label}
              </a>
            </li>
          ))}
        </ul>
        <div className="ml-auto hidden lg:block">
          <StatusLegend compact />
        </div>
      </nav>
    </header>
  );
}

export function Footer({ syncedAt, repo }: { syncedAt: string; repo: string }) {
  return (
    <footer className="border-t border-line bg-ink-950 py-14">
      <div className="mx-auto grid max-w-7xl gap-10 px-6 md:grid-cols-[1.4fr_1fr]">
        <div>
          <p className="font-display text-2xl font-semibold text-white">
            EARN<span className="text-accent">.</span>
          </p>
          <p className="mt-3 max-w-lg text-sm leading-relaxed text-slate-400">
            Research prototype for a student project. It shows real measurements where they exist and
            labels everything else. It is not a medical product and makes no clinical claims.
          </p>
          <a href={repo} className="mt-4 inline-block font-mono text-xs text-accent hover:underline">
            {repo.replace("https://", "")}
          </a>
        </div>
        <div className="space-y-3 text-sm">
          <div className="flex items-start gap-3">
            <StatusTag status="verified" />
            <p className="text-slate-400">Real and measured, computed from our own results files.</p>
          </div>
          <div className="flex items-start gap-3">
            <StatusTag status="baseline" />
            <p className="text-slate-400">A working method we compare against, not our contribution.</p>
          </div>
          <div className="flex items-start gap-3">
            <StatusTag status="in-progress" />
            <p className="text-slate-400">Our proposed method: specified, not yet validated.</p>
          </div>
          <div className="flex items-start gap-3">
            <SampleBadge />
            <p className="text-slate-400">Placeholder values for layout, never results.</p>
          </div>
          <p className="pt-2 font-mono text-[11px] text-slate-600">Data last synced {syncedAt}.</p>
        </div>
      </div>
    </footer>
  );
}
