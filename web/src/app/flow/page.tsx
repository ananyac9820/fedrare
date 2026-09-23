import type { Metadata } from "next";
import { FlowRail } from "@/components/flow/FlowRail";
import { PageHeader } from "@/components/ui/Motion";
import { Pager } from "@/components/ui/Pager";
import { StatusTag } from "@/components/ui/Status";

export const metadata: Metadata = {
  title: "The flow",
  description: "One federated round from the six hospitals to the ledger, with the real state of every step.",
};

export default function FlowPage() {
  return (
    <>
      <PageHeader
        eyebrow="00 · The flow"
        tags={
          <>
            <StatusTag status="verified" />
            <StatusTag status="failed" />
            <StatusTag status="in-progress" />
          </>
        }
        title="One round, from the hospitals to the ledger."
      >
        Ten steps, in the order the system actually runs them. Each carries its real state: what was
        measured, what came back negative, and what has only been designed. Scroll to follow the round through -
        every step links to the page with the detail behind it.
      </PageHeader>

      <section className="py-20 md:py-28">
        <FlowRail />
      </section>

      <Pager current="/flow" />
    </>
  );
}
