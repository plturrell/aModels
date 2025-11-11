import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';

const navItems = [
  {
    to: 'flows/',
    label: 'Flows',
    icon: (
      <svg
        className="h-4 w-4"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M7.5 4v9.5a4 4 0 1 0 4-4H5.5" />
      </svg>
    ),
  },
  {
    to: 'components/',
    label: 'Components',
    icon: (
      <svg
        className="h-4 w-4"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M12 2 2 7l10 5 10-5-10-5Z" />
        <path d="m2 17 10 5 10-5" />
        <path d="m2 12 10 5 10-5" />
      </svg>
    ),
  },
  {
    to: 'agentflow/',
    label: 'AgentFlow Catalog',
    icon: (
      <svg
        className="h-4 w-4"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M2 10c0 5.5228 4.4772 10 10 10s10-4.4772 10-10S17.5228 0 12 0" />
        <path d="M12 14v8" />
        <path d="m9 21 3 3 3-3" />
        <path d="m19 16 3 3-3 3" />
        <path d="m5 16-3 3 3 3" />
      </svg>
    ),
  },
  {
    to: 'mcp/',
    label: 'MCP',
    icon: (
      <svg
        className="h-4 w-4"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="12" r="5" />
        <path d="M12 1v3" />
        <path d="M12 20v3" />
        <path d="m4.22 4.22 2.12 2.12" />
        <path d="m17.66 17.66 2.12 2.12" />
        <path d="M1 12h3" />
        <path d="M20 12h3" />
        <path d="m4.22 19.78 2.12-2.12" />
        <path d="m17.66 6.34 2.12-2.12" />
      </svg>
    ),
  },
  {
    to: 'qa/',
    label: 'QA',
    icon: (
      <svg
        className="h-4 w-4"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M21 15a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2H3a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h6l3 3 3-3Z" />
        <path d="M8 10h.01" />
        <path d="M12 10h.01" />
        <path d="M16 10h.01" />
      </svg>
    ),
  },
];

const CollectionPage: React.FC = () => {
  return (
    <div className="flex h-full flex-col">
      <div className="main-page-nav-arrangement px-6">
        <div className="relative mx-auto flex w-full max-w-4xl items-center justify-center">
          <div className="flex w-full flex-wrap items-center justify-between gap-3">
            <span className="main-page-nav-title text-slate-900 dark:text-slate-100">
              Workspace
            </span>
            <nav
              aria-label="Primary"
              className="relative overflow-hidden backdrop-blur-xl bg-white/45 dark:bg-slate-900/40 border border-white/60 dark:border-slate-700/60 shadow-[0_18px_40px_rgba(15,23,42,0.16)] rounded-full px-2 py-1 transition-all duration-300"
            >
              <span className="nav-gradient" />
              <div className="pointer-events-none absolute inset-0 rounded-full border border-white/40 dark:border-slate-600/40 shadow-[inset_0_1px_rgba(255,255,255,0.35)]" />
              <ul className="flex flex-wrap items-center gap-1">
                {navItems.map((item) => (
                  <li key={item.to}>
                    <NavLink
                      to={item.to}
                      end={item.to === 'flows/'}
                      className={({ isActive }) => {
                        const base =
                          'group inline-flex items-center gap-1.5 rounded-full px-3.5 py-2 text-sm font-medium transition-all duration-200';
                        return isActive
                          ? `${base} relative nav-link-active bg-white text-slate-900 dark:bg-slate-700/80 dark:text-white`
                          : `${base} text-slate-600 hover:text-slate-900 hover:bg-white/60 dark:text-slate-300 dark:hover:text-white dark:hover:bg-slate-700/45`;
                      }}
                    >
                      <span className="text-slate-400 transition-colors duration-200 group-hover:text-slate-700 dark:group-hover:text-slate-200">
                        {item.icon}
                      </span>
                      {item.label}
                    </NavLink>
                  </li>
                ))}
              </ul>
            </nav>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        <Outlet />
      </div>
    </div>
  );
};

export default CollectionPage;
