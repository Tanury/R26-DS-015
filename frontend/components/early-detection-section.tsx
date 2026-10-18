import Image from "next/image";
import {
  AlertTriangle,
  Activity,
  Brain,
  Eye,
  FlaskConical,
  Mic,
  ScanSearch,
  Waves,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

/**
 * "Why Early Detection Matters" section for the home page.
 *
 * IMAGE SETUP — before this renders correctly, add these files under
 * /public/images/early-detection/ (create the folder if it doesn't exist):
 *
 *   mri.png            — example structural MRI slice (AD/PD/MS relevant view)
 *   datscan.png         — example DaTscan / SPECT striatal image
 *   oct.png              — example retinal OCT scan
 *   eeg.png              — example EEG waveform / band-power plot
 *
 * Use de-identified, anonymized sample images only — no real patient
 * identifiers, filenames, or metadata. Recommended size: roughly
 * 800x600px or similar 4:3 ratio, so the grid stays visually even.
 * If an image is missing, Next.js will show a broken-image icon in
 * that slot rather than failing the whole page.
 */

const diseases = [
  {
    name: "Alzheimer's Disease (AD)",
    icon: Brain,
    color: "bg-blue-700",
    description:
      "A progressive neurodegenerative disease and the most common cause of dementia, marked by the gradual loss of memory, thinking, and reasoning skills as neurons in the brain deteriorate.",
    earlySymptoms: [
      "Mild, recurring forgetfulness beyond normal aging",
      "Difficulty finding words or following conversations",
      "Trouble with planning or problem-solving",
    ],
    diagnosisNote:
      "Clinical diagnosis typically relies on cognitive testing after symptoms are already noticeable — by which point substantial hippocampal and cortical changes have often already occurred.",
  },
  {
    name: "Parkinson's Disease (PD)",
    icon: Activity,
    color: "bg-violet-700",
    description:
      "A progressive disorder of the nervous system caused by the loss of dopamine-producing neurons, primarily affecting movement but also linked to non-motor symptoms years before diagnosis.",
    earlySymptoms: [
      "Subtle tremor, often starting in one hand",
      "Reduced sense of smell (hyposmia)",
      "Sleep disturbances and mild rigidity or slowed movement",
    ],
    diagnosisNote:
      "Motor symptoms typically only become clinically obvious after a large proportion of dopamine-producing neurons have already been lost, since the brain compensates for early loss.",
  },
  {
    name: "Multiple Sclerosis (MS)",
    icon: Eye,
    color: "bg-emerald-700",
    description:
      "An autoimmune condition in which the immune system attacks the protective covering of nerve fibers, disrupting communication between the brain and the rest of the body.",
    earlySymptoms: [
      "Vision changes or eye pain in one eye",
      "Numbness, tingling, or weakness in limbs",
      "Fatigue and balance or coordination issues",
    ],
    diagnosisNote:
      "MS is often diagnosed only after an initial acute episode, though associated changes in the brain, spinal cord, and retina can sometimes be detected earlier through imaging.",
  },
];

const modalities = [
  {
    name: "Structural MRI",
    icon: ScanSearch,
    color: "bg-cyan-700",
    image: "/images/early-detection/mri.png",
    checks:
      "Regional brain volume and cortical thickness — atrophy patterns in the hippocampus and temporal lobe (AD), and white matter lesion load (MS).",
  },
  {
    name: "DaTscan (SPECT)",
    icon: Waves,
    color: "bg-violet-700",
    image: "/images/early-detection/datscan.png",
    checks:
      "Dopamine transporter density in the striatum — reduced binding is a specific marker of dopaminergic neuron loss in Parkinson's disease.",
  },
  {
    name: "Retinal OCT",
    icon: Eye,
    color: "bg-emerald-700",
    image: "/images/early-detection/oct.png",
    checks:
      "Retinal nerve fiber layer and ganglion cell thickness — the retina is an extension of the central nervous system, so thinning here can reflect neurodegeneration elsewhere.",
  },
  {
    name: "EEG",
    icon: Activity,
    color: "bg-blue-700",
    image: "/images/early-detection/eeg.png",
    checks:
      "Band-power distribution and signal complexity — abnormal slowing or connectivity patterns associated with early cognitive and neurological decline.",
  },
  {
    name: "Voice & Speech",
    icon: Mic,
    color: "bg-amber-700",
    image: null,
    checks:
      "Acoustic biomarkers — speech fluency, pause patterns, and articulation changes that can precede more obvious clinical symptoms.",
  },
  {
    name: "Clinical & Lab Biomarkers",
    icon: FlaskConical,
    color: "bg-rose-700",
    image: null,
    checks:
      "Genetic, metabolic, and laboratory values statistically associated with elevated neurological risk, used alongside imaging and functional markers.",
  },
];

export function EarlyDetectionSection() {
  return (
    <div className="mt-10">
      <Card>
        <CardContent className="p-6 sm:p-8">
          <h2 className="text-2xl font-bold sm:text-3xl">Why Early Detection Matters</h2>
          <p className="mt-4 max-w-3xl leading-7 text-slate-600">
            Alzheimer&apos;s disease, Parkinson&apos;s disease, and multiple sclerosis are
            each typically diagnosed only after symptoms are already noticeable to the
            person or their doctor — often well after the underlying neurological changes
            began. Our research explores whether combining structural, functional, and
            biomedical signals can help surface risk earlier, before symptoms make it
            clinically obvious.
          </p>
        </CardContent>
      </Card>

      {/* Disease overview cards */}
      <div className="mt-6 grid gap-6 md:grid-cols-3">
        {diseases.map(({ name, icon: Icon, color, description, earlySymptoms, diagnosisNote }) => (
          <Card key={name}>
            <CardContent className="flex h-full flex-col p-6">
              <div className={`grid size-11 place-items-center rounded-lg text-white ${color}`}>
                <Icon className="size-5" />
              </div>
              <h3 className="mt-4 text-lg font-bold">{name}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-600">{description}</p>

              <div className="mt-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Early symptoms often overlooked
                </p>
                <ul className="mt-2 space-y-1.5">
                  {earlySymptoms.map((symptom) => (
                    <li key={symptom} className="text-sm leading-6 text-slate-600">
                      • {symptom}
                    </li>
                  ))}
                </ul>
              </div>

              <p className="mt-4 text-xs leading-5 text-slate-500">{diagnosisNote}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Multimodal approach */}
      <Card className="mt-6">
        <CardContent className="p-6 sm:p-8">
          <h2 className="text-2xl font-bold sm:text-3xl">Our Multimodal Approach</h2>
          <p className="mt-3 max-w-3xl leading-7 text-slate-600">
            Rather than waiting for a single, obvious symptom, our platform draws on
            several independent evidence sources — each capturing a different signal
            associated with disease-specific neurological change.
          </p>

          <div className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {modalities.map(({ name, icon: Icon, color, image, checks }) => (
              <div key={name} className="overflow-hidden rounded-lg border border-slate-200">
                {image ? (
                  <div className="relative h-40 w-full bg-slate-100">
                    <Image
                      src={image}
                      alt={`Example ${name} scan`}
                      fill
                      className="object-cover"
                    />
                  </div>
                ) : (
                  <div className={`flex h-40 w-full items-center justify-center text-white ${color}`}>
                    <Icon className="size-10 opacity-90" />
                  </div>
                )}
                <div className="p-4">
                  <div className="flex items-center gap-2">
                    <div className={`grid size-7 shrink-0 place-items-center rounded-md text-white ${color}`}>
                      <Icon className="size-4" />
                    </div>
                    <h3 className="font-semibold">{name}</h3>
                  </div>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{checks}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Disclaimer */}
      <div className="mt-6 flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4 sm:p-5">
        <AlertTriangle className="mt-0.5 size-5 shrink-0 text-amber-700" />
        <p className="text-sm leading-6 text-amber-900">
          This platform is a research prototype intended to explore early-detection
          methods and is not a diagnostic tool. It is not a substitute for professional
          medical evaluation. Anyone with health concerns should consult a qualified
          healthcare provider.
        </p>
      </div>
    </div>
  );
}