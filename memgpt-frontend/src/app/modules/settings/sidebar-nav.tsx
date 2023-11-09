"use client"

import { Link } from "@tanstack/react-router"

import { cn } from "@memgpt/utils"
import { buttonVariants } from "@memgpt/components/button"
import { SidebarNavItem } from './settings';
import { settingsRoute } from './settings.routes';

interface SidebarNavProps extends React.HTMLAttributes<HTMLElement> {
  items: SidebarNavItem[]
}

export function SidebarNav({ className, items, ...props }: SidebarNavProps) {
  return (
    <nav
      className={cn(
        "flex space-x-2 lg:flex-col lg:space-x-0 lg:space-y-1",
        className
      )}
      {...props}
    >
      {items.map((item,i) => (
        // @ts-ignore
        <Link
          key={i}
          from={settingsRoute.id}
          to={item.to}
          className={cn(
            buttonVariants({ variant: "ghost" }),
            "hover:bg-transparent hover:underline",
            "data-[status=active]:bg-muted data-[status=active]:hover:bg-muted data-[status=active]:hover:no-underline",
            "justify-start"
          )}
        >
          {item.title}
        </Link>
      ))}
    </nav>
  )
}
