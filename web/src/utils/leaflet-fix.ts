/**
 * Leaflet 默认图标路径修复（Vite 环境下需手动指定）
 *
 * Cockpit 和 MapView 页面共享，使用 fixed 标志位防止重复执行
 * 同时按需加载 Leaflet CSS，避免非地图页面加载无关样式
 */

import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

let fixed = false;

export function fixLeafletIcons() {
  if (fixed) return;
  // @ts-expect-error vite leaflet icon path fix
  delete L.Icon.Default.prototype._getIconUrl;
  L.Icon.Default.mergeOptions({
    iconRetinaUrl: markerIcon2x,
    iconUrl: markerIcon,
    shadowUrl: markerShadow,
  });
  fixed = true;
}
