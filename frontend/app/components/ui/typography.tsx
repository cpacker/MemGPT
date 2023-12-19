import { cn } from "@/app/lib/utils";

type HeadingProps = React.DetailedHTMLProps<
  React.HTMLAttributes<HTMLHeadingElement>,
  HTMLHeadingElement
>;

type ParagraphProps = React.DetailedHTMLProps<
  React.HTMLAttributes<HTMLParagraphElement>,
  HTMLDivElement
>;

type LinkProps = React.DetailedHTMLProps<
  React.AnchorHTMLAttributes<HTMLAnchorElement>,
  HTMLAnchorElement
>;

function H1(props: HeadingProps) {
  const { className, children, ...rest } = props;
  return (
    <h1
      className={cn(
        "scroll-m-20 text-4xl font-extrabold tracking-tight",
        className,
      )}
      {...rest}
    >
      {children}
    </h1>
  );
}

function H2(props: HeadingProps) {
  const { className, children, ...rest } = props;
  return (
    <h2
      className={cn(
        "scroll-m-20 text-3xl font-semibold tracking-tight",
        className,
      )}
      {...rest}
    >
      {children}
    </h2>
  );
}

function H3(props: HeadingProps) {
  const { className, children, ...rest } = props;
  return (
    <h3
      className={cn(
        "scroll-m-20 text-2xl font-semibold tracking-tight",
        className,
      )}
      {...rest}
    >
      {children}
    </h3>
  );
}

function H4(props: HeadingProps) {
  const { className, children, ...rest } = props;
  return (
    <h4
      className={cn(
        "scroll-m-20 text-xl font-semibold tracking-tight",
        className,
      )}
      {...rest}
    >
      {children}
    </h4>
  );
}

function P(props: ParagraphProps) {
  const { className, children, ...rest } = props;
  return (
    <p className={cn("leading-7", className)} {...rest}>
      {children}
    </p>
  );
}

function Link(props: LinkProps) {
  const { className, children, ...rest } = props;
  return (
    <a
      className={cn(
        "font-medium text-primary underline underline-offset-4",
        className,
      )}
      {...rest}
    >
      {children}
    </a>
  );
}

const Typography = {
  H1,
  H2,
  H3,
  H4,
  P,
  Link,
};

export default Typography;
