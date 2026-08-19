export function PageHeader({ title, description }: { title: string; description: string }) {
  return (
    <div className="mb-7">
      <h1 className="page-title text-slate-950">{title}</h1>
      <p className="mt-2 max-w-3xl text-base leading-7 text-slate-600 sm:text-lg">{description}</p>
    </div>
  );
}
