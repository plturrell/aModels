#!/usr/bin/env python3
"""Generate SGMI flow data for the browser shell visualisations."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / 'data/training/sgmi/json_with_changes.json'
TARGET_PATH = Path(__file__).resolve().parents[1] / 'ui' / 'src' / 'data' / 'sgmi_flow.json'


def _load_source() -> dict:
    with SOURCE_PATH.open() as handle:
        return json.load(handle)


def _walk(obj, meta: dict, edges: Counter):
    stack = [('', obj)]
    while stack:
        parent_key, current = stack.pop()
        if isinstance(current, dict):
            if 'eventsToAdd' in current or 'EventsToAdd' in current:
                events = current.get('eventsToAdd') or current.get('EventsToAdd')
                for event in events.get('Events', []):
                    evt = event.get('Event') if isinstance(event, dict) else event
                    if isinstance(evt, str) and '-TO-' in evt:
                        source, target = evt.split('-TO-', 1)
                        edges[(source.strip(), target.strip())] += 1
            if 'FileName' in current or 'Command' in current or str(current.get('Type', '')).startswith(
                'Job'
            ):
                job_name = current.get('FileName') or parent_key
                job_name = parent_key or job_name
                if isinstance(job_name, str):
                    info = meta.setdefault(job_name, {})
                    info.setdefault('type', current.get('Type'))
                    info.setdefault(
                        'application',
                        current.get('Application')
                        or current.get('SubApplication')
                        or current.get('SiteStandard'),
                    )
                    if current.get('Host'):
                        info['host'] = current['Host']
                    if current.get('Command'):
                        info['command'] = current['Command']
                    if current.get('RunAs'):
                        info['run_as'] = current['RunAs']
                    if current.get('Priority'):
                        info['priority'] = current['Priority']
                    if current.get('Description'):
                        info['description'] = current['Description']
                    if current.get('ControlmServer'):
                        info['controlm_server'] = current['ControlmServer']
                    if current.get('SubApplication'):
                        info['sub_application'] = current['SubApplication']
                    if current.get('SiteStandard'):
                        info['site_standard'] = current['SiteStandard']
                    if current.get('DaysKeepActive'):
                        info['days_keep_active'] = current['DaysKeepActive']
                    if current.get('RerunLimit'):
                        info['rerun_limit'] = current['RerunLimit']
                    if current.get('When'):
                        info['schedule'] = current['When']
                    if current.get('Variables'):
                        info['variables'] = current['Variables']
            for key, value in current.items():
                stack.append((key, value))
        elif isinstance(current, list):
            for item in current:
                stack.append((parent_key, item))


def build_dataset() -> dict:
    meta: dict[str, dict] = {}
    edges: Counter[tuple[str, str]] = Counter()
    _walk(_load_source(), meta, edges)

    nodes = []
    node_index: dict[str, int] = {}
    for idx, name in enumerate(sorted({name for pair in edges for name in pair})):
        node_index[name] = idx
        info = meta.get(name, {})
        nodes.append(
            {
                'id': name,
                'type': info.get('type'),
                'application': info.get('application'),
                'meta': info,
            }
        )

    links = [
        {'source': source, 'target': target, 'value': count}
        for (source, target), count in edges.most_common()
    ]

    network_nodes = [
        {
            'id': node['id'],
            'group': node.get('application') or 'unknown',
            'type': node.get('type'),
            'meta': node.get('meta', {}),
        }
        for node in nodes
    ]
    network_links = [
        {
            'source': link['source'],
            'target': link['target'],
            'value': link['value'],
        }
        for link in links
    ]

    return {
        'sankey': {'nodes': nodes, 'links': links},
        'network': {'nodes': network_nodes, 'links': network_links},
    }


def main() -> None:
    dataset = build_dataset()
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_PATH.open('w') as handle:
        json.dump(dataset, handle, indent=2)
    print(f'Wrote {TARGET_PATH} (nodes={len(dataset["sankey"]["nodes"])} links={len(dataset["sankey"]["links"])}).')


if __name__ == '__main__':
    main()
