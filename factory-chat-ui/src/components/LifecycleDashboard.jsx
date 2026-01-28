// src/components/LifecycleDashboard.jsx
import React, { useMemo, useState } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis, AreaChart, Area
} from 'recharts';
import {
    Activity, Clock, MapPin, Package, Layers, Search,
    UploadCloud, ArrowLeft, Timer, MousePointer2, AlertTriangle, FileText, X
} from 'lucide-react';

// 炫彩配色
const COLORS = ['#6366f1', '#ec4899', '#8b5cf6', '#14b8a6', '#f59e0b', '#3b82f6'];

const LifecycleDashboard = ({ isOpen, onClose }) => {
    const [data, setData] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [searchTerm, setSearchTerm] = useState('');

    // --- 上传文件 ---
    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        setIsLoading(true);
        setError(null);
        const formData = new FormData();
        formData.append('file', file);

        try {
            // 注意：确保端口号与后端一致
            const response = await fetch('http://localhost:8000/api/upload_lifecycle', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();
            if (result.data) {
                setData(result.data);
            } else {
                setError(result.error || "文件解析失败，请检查格式");
            }
        } catch (err) {
            setError("上传失败，请检查后端服务是否启动");
        } finally {
            setIsLoading(false);
        }
    };

    // 如果未打开，不渲染
    if (!isOpen) return null;

    // --- 界面 A: 上传文件 ---
    if (!data) {
        return (
            <div className="fixed inset-0 z-50 bg-slate-100 flex flex-col animate-in fade-in duration-200">
                {/* 顶部栏 */}
                <div className="bg-white p-4 shadow-sm flex justify-between items-center">
                    <h1 className="text-xl font-bold text-slate-800 flex items-center gap-2">
                        <Activity className="text-blue-600" /> 零件生命周期可视化面板
                    </h1>
                    <button onClick={onClose} className="p-2 hover:bg-slate-100 rounded-full transition-colors">
                        <X size={24} className="text-slate-500" />
                    </button>
                </div>

                {/* 上传区域 */}
                <div className="flex-1 flex flex-col items-center justify-center p-8">
                    <div className="bg-white p-12 rounded-3xl shadow-xl text-center max-w-2xl w-full border border-slate-200">
                        <div className="mb-6 inline-block p-4 bg-blue-50 rounded-full">
                            <UploadCloud size={64} className="text-blue-500" />
                        </div>
                        <h2 className="text-2xl font-bold text-slate-800 mb-2">上传生产流转数据</h2>
                        {/* 修改提示文本 */}
                        <p className="text-slate-500 mb-8">请上传包含唯一编号、工位、坐标和耗时信息的表格文件</p>
                        <label className={`flex flex-col items-center justify-center w-full h-48 border-2 border-dashed rounded-xl cursor-pointer transition-all ${isLoading ? 'bg-slate-50 border-slate-300' : 'bg-blue-50/50 border-blue-300 hover:border-blue-500 hover:bg-blue-50'}`}>
                            {isLoading ? (
                                <div className="flex flex-col items-center">
                                    <div className="animate-spin rounded-full h-10 w-10 border-4 border-blue-200 border-t-blue-600 mb-4"></div>
                                    <span className="text-slate-500 font-medium">正在分析数据...</span>
                                </div>
                            ) : (
                                <div className="flex flex-col items-center">
                                    <span className="text-blue-600 font-bold text-lg mb-2">点击选择文件</span>
                                    {/* 修改支持格式提示 */}
                                    <span className="text-slate-400 text-sm">支持 .xlsx / .csv 格式</span>
                                </div>
                            )}
                            {/* 修改 accept 属性，增加 .xlsx 和 .xls */}
                            <input type="file" accept=".csv, .xlsx, .xls" className="hidden" onChange={handleFileUpload} disabled={isLoading} />
                        </label>

                        {error && (
                            <div className="mt-6 p-4 bg-red-50 text-red-600 rounded-lg flex items-center justify-center gap-2 border border-red-100">
                                <AlertTriangle size={18} /> {error}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    }

    // --- 数据计算 ---
    // --- 1. 数据完整度计算逻辑 ---
    const calculateIntegrity = () => {
        if (!data || data.length === 0) return 0;

        let totalCells = 0;
        let filledCells = 0;

        data.forEach(row => {
            Object.values(row).forEach(val => {
                totalCells++;
                // 判断有效值：不是 null, undefined, 且不是空字符串
                // 注意：数字 0 是有效数据，不能被排除
                if (val !== null && val !== undefined && val !== "") {
                    filledCells++;
                }
            });
        });

        return totalCells === 0 ? 0 : ((filledCells / totalCells) * 100).toFixed(1);
    };

    const dataIntegrity = calculateIntegrity();
    const totalTasks = data.length;
    // 计算总耗时(需处理可能的非数字字符)
    const avgTime = (data.reduce((sum, item) => sum + (Number(item['总耗时(分钟)']) || 0), 0) / totalTasks).toFixed(1);
    const stationsCount = new Set(data.map(i => i['最新工位 (Station)'])).size;

    // 图表数据1: 工位排行
    const stationData = Object.entries(data.reduce((acc, item) => {
        const s = item['最新工位 (Station)'] || '未知';
        acc[s] = (acc[s] || 0) + 1;
        return acc;
    }, {})).map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count).slice(0, 10);

    // 图表数据2: 类型占比
    const typeData = Object.entries(data.reduce((acc, item) => {
        const t = item['零件类型 (Type)'] || '其他';
        acc[t] = (acc[t] || 0) + 1;
        return acc;
    }, {})).map(([name, value]) => ({ name, value }));

    // 图表数据3: 车间地图
    const mapData = data.map(item => ({
        x: Number(item['坐标 X']) || 0,
        y: Number(item['坐标 Y']) || 0,
        z: Number(item['总耗时(分钟)']) || 10,
        name: item['唯一编号 (Unique ID)'],
        station: item['最新工位 (Station)']
    }));

    // 表格过滤
    const filteredData = data.filter(item =>
        JSON.stringify(item).toLowerCase().includes(searchTerm.toLowerCase())
    );

    // --- 界面 B: 数据看板 ---
    return (
        <div className="fixed inset-0 z-50 bg-slate-50 flex flex-col overflow-hidden animate-in slide-in-from-bottom-10 duration-300">
            {/* 顶部导航 */}
            <div className="bg-white px-6 py-4 shadow-sm border-b border-slate-200 flex justify-between items-center shrink-0">
                <div className="flex items-center gap-4">
                    <button onClick={() => setData(null)} className="p-2 hover:bg-slate-100 rounded-full text-slate-500" title="返回上传">
                        <ArrowLeft size={24} />
                    </button>
                    <div>
                        <h1 className="text-xl font-bold text-slate-800">生产监测看板</h1>
                        <div className="flex items-center gap-2 text-xs text-slate-500">
                            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                            系统在线 · 共 {totalTasks} 条数据
                        </div>
                    </div>
                </div>
                <button onClick={onClose} className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg font-medium transition-colors">
                    退出面板
                </button>
            </div>

            {/* 滚动内容区 */}
            <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
                <div className="max-w-7xl mx-auto space-y-6">

                    {/* KPI 指标 */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                        <KPICard title="总任务数" value={totalTasks} unit="个" icon={<Package size={24} />} color="blue" />
                        <KPICard title="平均耗时" value={avgTime} unit="min" icon={<Clock size={24} />} color="indigo" />
                        <KPICard title="活跃工位" value={stationsCount} unit="个" icon={<MapPin size={24} />} color="emerald" />
                        <KPICard title="数据完整度" value={dataIntegrity} unit="%" icon={<FileText size={24} />} color="violet" />
                    </div>
                    {/* 图表第一行 */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* 散点图 (2/3) */}
                        <div className="lg:col-span-2 bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
                            <h3 className="font-bold text-slate-700 mb-6 flex items-center gap-2">
                                <MousePointer2 className="text-blue-500" /> 车间物流位置热力图 (坐标 XY)
                            </h3>
                            <div className="h-[350px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis type="number" dataKey="x" name="X轴" unit="mm" />
                                        <YAxis type="number" dataKey="y" name="Y轴" unit="mm" />
                                        <ZAxis type="number" dataKey="z" range={[50, 400]} name="耗时" unit="min" />
                                        <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<CustomTooltip />} />
                                        <Scatter name="零件" data={mapData} fill="#8884d8">
                                            {mapData.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
                                        </Scatter>
                                    </ScatterChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* 饼图 (1/3) */}
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
                            <h3 className="font-bold text-slate-700 mb-6 flex items-center gap-2">
                                <Activity className="text-pink-500" /> 零件类型分布
                            </h3>
                            <div className="h-[350px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie data={typeData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                                            {typeData.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
                                        </Pie>
                                        <Tooltip />
                                        <Legend layout="vertical" verticalAlign="bottom" align="center" />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>

                    {/* 图表第二行: 柱状图 */}
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
                        <h3 className="font-bold text-slate-700 mb-6 flex items-center gap-2">
                            <Layers className="text-indigo-500" /> 工位任务负载 Top 10
                        </h3>
                        <div className="h-[300px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={stationData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                    <XAxis dataKey="name" />
                                    <YAxis />
                                    <Tooltip cursor={{ fill: '#f8fafc' }} />
                                    <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} barSize={40}>
                                        {stationData.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* 数据明细表 */}
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
                        <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
                            <h3 className="font-bold text-slate-700 flex items-center gap-2">
                                <FileText className="text-slate-400" /> 数据明细
                            </h3>
                            <div className="relative w-64">
                                <input
                                    type="text"
                                    placeholder="搜索唯一编号、料框号..."
                                    className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                />
                                <Search className="absolute left-3 top-2.5 text-slate-400" size={16} />
                            </div>
                        </div>
                        <div className="overflow-x-auto max-h-[500px]">
                            <table className="min-w-full text-sm text-left">
                                <thead className="bg-slate-50 text-slate-500 font-medium sticky top-0 z-10">
                                    <tr>
                                        <th className="px-6 py-4">唯一编号</th>
                                        <th className="px-6 py-4">零件类型</th>
                                        <th className="px-6 py-4">料框号</th>
                                        <th className="px-6 py-4">任务号</th>
                                        <th className="px-6 py-4">最新工位</th>
                                        <th className="px-6 py-4">坐标 (X, Y)</th>
                                        <th className="px-6 py-4 text-right">总耗时(分钟)</th>
                                        <th className="px-6 py-4 text-right">开始时间</th>
                                        <th className="px-6 py-4 text-right">结束时间</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {filteredData.slice(0, 50).map((item, idx) => (
                                        <tr key={idx} className="hover:bg-blue-50/30 transition-colors">
                                            <td className="px-6 py-3 font-mono text-slate-500">{item['唯一编号 (Unique ID)']}</td>
                                            <td className="px-6 py-3">
                                                {/* 添加 whitespace-nowrap 防止换行 */}
                                                <span className="bg-slate-100 px-2 py-1 rounded text-xs text-slate-600 whitespace-nowrap">
                                                    {item['零件类型 (Type)']}
                                                </span>
                                            </td>
                                            <td className="px-6 py-3 font-medium text-slate-800">{item['料框号 (Frame Code)']}</td>
                                            <td className="px-6 py-3 font-medium text-slate-800">{item['任务号 (Task No)']}</td>
                                            <td className="px-6 py-3 text-slate-600">{item['最新工位 (Station)']}</td>
                                            <td className="px-6 py-3 font-mono text-xs text-slate-400">({item['坐标 X']}, {item['坐标 Y']})</td>
                                            <td className="px-6 py-3 text-right font-bold text-blue-600">{item['总耗时(分钟)']}</td>
                                            <td className="px-6 py-3 text-right font-bold text-blue-600">
                                                {item['开始时间'] ? String(item['开始时间']).split('.')[0] : '-'}
                                            </td>

                                            <td className="px-6 py-3 text-right font-bold text-blue-600">
                                                {item['结束时间'] ? String(item['结束时间']).split('.')[0] : '-'}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
};

// --- 子组件 ---
const KPICard = ({ title, value, unit, icon, color }) => (
    <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm flex items-center gap-4">
        <div className={`p-4 rounded-xl bg-${color}-50 text-${color}-600`}>{icon}</div>
        <div>
            <p className="text-slate-400 text-sm font-medium">{title}</p>
            <div className="flex items-baseline gap-1">
                <span className="text-2xl font-bold text-slate-800">{value}</span>
                <span className="text-xs text-slate-400">{unit}</span>
            </div>
        </div>
    </div>
);

const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        return (
            <div className="bg-white p-3 border border-slate-100 rounded-lg shadow-xl text-xs z-50">
                <p className="font-bold mb-1">{data.name}</p>
                <p className="text-slate-500">工位: {data.station}</p>
                <p className="text-blue-600 font-bold">耗时: {data.z} min</p>
            </div>
        );
    }
    return null;
};

export default LifecycleDashboard;