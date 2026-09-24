import type { Metadata } from "next";
import { DemoConsole } from "@/components/demo/DemoConsole";
import { Card, PillLink } from "@/components/ui/Buttons";
import { PageHeader, Reveal, Section } from "@/components/ui/Motion";

export const metadata: Metadata = {
  title: "Live demo",
  description:
    "Run the federated model this project trained on held-out Fed-ISIC2019 images, locally.",
};

const STEPS = [
  {
    n: "01",
    title: "Six hospitals, one model",
    body: "The classifier head was trained by FedAvg across the six real Fed-ISIC2019 centres. No hospital sent an image anywhere; each sent only its updated weights.",
  },
  {
    n: "02",
    title: "Frozen features, trained head",
    body: "An ImageNet DenseNet-121 turns each image into 1,024 numbers. Only the final layer is federated — which is what makes the whole study runnable on a laptop.",
  },
  {
    n: "03",
    title: "Held-out images",
    body: "The eight images on this page come from the test split, chosen by class and hospital before the model was run. Two are Dermatofibroma and two are Vascular lesion: the rare classes this project is about.",
  },
];

export default function DemoPage() {
  return (
    <>
      <PageHeader eyebrow="Live demo" title="Run the model on a real image.">
        This page talks to a Python server on your own machine. It loads the model the six
        hospitals trained together, runs one dermoscopy image through it and shows what it
        predicts across all eight classes — including when it gets the answer wrong.
      </PageHeader>

      <Section>
        <DemoConsole />
      </Section>

      <Section
        tone="sand"
        label="What you are looking at"
        title="The same model the results pages report."
        intro="Nothing here is a mock-up: the predictions come from the FedAvg run on the natural split, loaded from a saved checkpoint."
      >
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-6 md:grid-cols-3">
          {STEPS.map((s, i) => (
            <Reveal key={s.n} delay={i * 0.08}>
              <Card className="h-full">
                <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-faint">{s.n}</p>
                <h3 className="mt-5 font-display text-2xl text-ink">{s.title}</h3>
                <p className="mt-4 text-sm leading-relaxed text-muted">{s.body}</p>
              </Card>
            </Reveal>
          ))}
        </div>

        <Reveal className="mt-10">
          <Card>
            <h3 className="font-display text-2xl text-ink">Read it honestly</h3>
            <p className="mt-4 max-w-3xl text-sm leading-relaxed text-muted">
              This is a research prototype on a public dataset, not a medical device, and it must
              not be used on anyone&apos;s skin. The model is right about two times in five across
              the eight classes — better than chance, far below clinical use — so the true label is
              always shown beside the prediction rather than only when it agrees. Uploaded photos
              are further from the dataset than the samples are: ordinary phone pictures are not
              dermoscopy images, and the model will answer confidently anyway.
            </p>
          </Card>
        </Reveal>

        <Reveal className="mt-12 flex flex-wrap gap-3">
          <PillLink href="/results">See how it was measured</PillLink>
          <PillLink href="/flow" variant="outline">
            The whole system, end to end
          </PillLink>
        </Reveal>
      </Section>
    </>
  );
}
