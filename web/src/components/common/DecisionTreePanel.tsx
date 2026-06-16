/**
 * 预警分级决策树 — 可视化组件
 *
 * 递归渲染决策树节点，展示从检测帧输入到最终预警等级的完整判定路径。
 * 纯展示组件，无交互状态。
 */

import { memo, useEffect, useState } from 'react';
import { Spin, Empty } from 'antd';
import { fetchDecisionTree } from '../../services/api';
import type { TreeNode, Branch } from '../../services/api';
import './DecisionTreePanel.css';

// ─── 叶子节点 ───
const LeafNode = memo(function LeafNode({
  type,
  label,
}: {
  type: string;
  label: string;
}) {
  const className = `tree-leaf ${type}`;
  return <div className={className}>{label}</div>;
});

// ─── 决策节点 (递归) ───
const DecisionNode = memo(function DecisionNode({
  label,
  branches,
}: {
  label: string;
  branches: Branch[];
}) {
  return (
    <div className="tree-subtree">
      <div className="tree-node decision">{label}</div>
      <div className="tree-arrow">▼</div>
      <div className="tree-branch">
        {branches.map((b, i) => (
          <div className="tree-branch-item" key={i}>
            <div className="tree-label">{b.label}</div>
            {b.result ? (
              <LeafNode type={b.result.type} label={b.result.label} />
            ) : b.node ? (
              <DecisionNode label={b.node.label} branches={b.node.branches || []} />
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
});

// ─── 主组件 ───
function DecisionTreePanel() {
  const [tree, setTree] = useState<TreeNode | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetchDecisionTree()
      .then((data) => {
        if (!cancelled) setTree(data);
      })
      .catch(() => { /* ignore */ })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) return <Spin size="small" style={{ display: 'block', margin: '24px auto' }} />;
  if (!tree) return <Empty description="无法加载决策树" />;

  // 根节点的 children 是顶层分支 (conf decision + green leaf)
  const confDecision = tree.children?.[0]; // "最高置信度 max_conf ?"
  const greenLeaf = tree.children?.[1];    // "< 0.30 → 正常"

  return (
    <div className="decision-tree-container">
      {/* ROOT */}
      <div className="tree-node root">{tree.label}</div>
      <div className="tree-arrow">▼</div>

      {/* Confidence decision */}
      {confDecision && confDecision.branches && (
        <DecisionNode
          label={confDecision.label}
          branches={confDecision.branches}
        />
      )}

      {/* < 0.30 → green */}
      {greenLeaf && (
        <div style={{ textAlign: 'center', marginTop: '0.5rem' }}>
          <div className="tree-label">&lt; 0.30</div>
          <LeafNode type={greenLeaf.type} label={greenLeaf.label} />
        </div>
      )}
    </div>
  );
}

export default DecisionTreePanel;
