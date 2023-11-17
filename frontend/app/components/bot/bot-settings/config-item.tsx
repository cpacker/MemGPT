export default function ConfigItem(props: {
  title: string;
  subTitle?: string;
  children: JSX.Element;
}) {
  return (
    <div className="flex justify-between items-center min-h-[40px] px-3 py-5 gap-4 first:pt-0 last:pb-0">
      <div className="w-1/2">
        <div className="font-medium text-sm">{props.title}</div>
        <div className="text-xs text-muted-foreground">{props.subTitle}</div>
      </div>
      <div className="text-sm">{props.children}</div>
    </div>
  );
}
