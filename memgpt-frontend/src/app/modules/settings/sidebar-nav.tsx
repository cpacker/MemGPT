'use client';

import { cn } from '@memgpt/utils';
import { buttonVariants } from '@memgpt/components/button';
import { SidebarNavItem } from './settings';
import { NavLink } from 'react-router-dom';

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
        <NavLink
          relative="path"
          key={i}
          to={item.to}
          className={cn(
            buttonVariants({ variant: "ghost" }),
            "hover:bg-transparent hover:underline",
            "[&.active]:bg-muted [&.active]:hover:bg-muted [&.active]:hover:no-underline",
            "justify-start"
          )}
        >
          {item.title}
        </NavLink>
      ))}
    </nav>
  )
}
