import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

export function BackButton({
  href,
  label,
  className = "mb-5",
}: {
  href: string;
  label: string;
  className?: string;
}) {
  return (
    <Button className={className} variant="outline" size="sm" asChild>
      <Link href={href}>
        <ArrowLeft className="size-4" />
        {label}
      </Link>
    </Button>
  );
}
