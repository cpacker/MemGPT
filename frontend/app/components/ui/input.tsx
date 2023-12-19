import * as React from "react";

import { cn } from "@/app/lib/utils";

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

export interface InputRangeProps {
  onChange: React.ChangeEventHandler<HTMLInputElement>;
  title?: string;
  value: number | string;
  className?: string;
  min: string;
  max: string;
  step: string;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);
Input.displayName = "Input";

function InputRange(props: InputRangeProps) {
  const { className, title, value, ...rest } = props;
  return (
    <div
      className={cn(
        "rounded-lg border-2 border-muted p-2 text-sm flex items-center gap-2",
        className,
      )}
    >
      <span>{title || value}</span>
      <Input
        className="p-0 accent-primary"
        type="range"
        title={title}
        value={value}
        {...rest}
      ></Input>
    </div>
  );
}

export { Input, InputRange };
