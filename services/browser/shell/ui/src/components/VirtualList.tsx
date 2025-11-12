/**
 * Virtual List Component
 * 
 * Efficiently renders large lists using react-window
 * Only renders visible items for performance
 */

// @ts-ignore - react-window provides its own types
import { FixedSizeList } from 'react-window';

interface VirtualListProps<T> {
  items: T[];
  height: number;
  itemHeight: number;
  width?: string | number;
  renderItem: (item: T, index: number, style: React.CSSProperties) => React.ReactNode;
  overscanCount?: number;
}

interface RowProps {
  index: number;
  style: React.CSSProperties;
}

export function VirtualList<T>({
  items,
  height,
  itemHeight,
  width = '100%',
  renderItem,
  overscanCount = 5,
}: VirtualListProps<T>) {
  const Row = ({ index, style }: RowProps) => (
    <div style={style}>
      {renderItem(items[index], index, style)}
    </div>
  );

  return (
    <FixedSizeList
      height={height}
      itemCount={items.length}
      itemSize={itemHeight}
      width={width}
      overscanCount={overscanCount}
    >
      {Row}
    </FixedSizeList>
  );
}

/**
 * Example usage:
 * 
 * <VirtualList
 *   items={nodes}
 *   height={600}
 *   itemHeight={50}
 *   renderItem={(node, index, style) => (
 *     <ListItem button onClick={() => handleClick(node)}>
 *       <ListItemText primary={node.label} />
 *     </ListItem>
 *   )}
 * />
 */
