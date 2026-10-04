export type WorkflowNode = {
  id: string;
  type: string;
  config: Record<string, string | number | boolean>;
  transitions: Record<string, string>;
};

export type WorkflowDraft = {
  name: string;
  entry: string;
  nodes: WorkflowNode[];
  max_steps: number;
};

export type SavedWorkflow = WorkflowDraft & {
  id: string;
  version: number;
  updated_at: string;
};

export type ConfigProperty = {
  title?: string;
  description?: string;
  type?: "string" | "integer" | "number" | "boolean";
  default?: string | number | boolean;
  minimum?: number;
  maximum?: number;
  maxLength?: number;
};

export type NodeType = {
  type: string;
  label: string;
  kind: "agent" | "tool";
  description: string;
  outputs: string[];
  config_schema: { properties?: Record<string, ConfigProperty> };
};

export function copyDraft(workflow: SavedWorkflow): WorkflowDraft {
  return {
    name: workflow.name,
    entry: workflow.entry,
    max_steps: workflow.max_steps,
    nodes: workflow.nodes.map(node => ({
      id: node.id, type: node.type, config: { ...node.config }, transitions: { ...node.transitions },
    })),
  };
}

export function unreachableNodeIds(draft: WorkflowDraft): string[] {
  const nodes = new Map(draft.nodes.map(node => [node.id, node]));
  const reached = new Set<string>();
  const pending = [draft.entry];
  while (pending.length) {
    const id = pending.pop()!;
    if (reached.has(id) || !nodes.has(id)) continue;
    reached.add(id);
    pending.push(...Object.values(nodes.get(id)!.transitions));
  }
  return draft.nodes.filter(node => !reached.has(node.id)).map(node => node.id);
}
