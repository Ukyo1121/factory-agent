// src/components/LifecycleDashboard.jsx
import React, { useState, useEffect } from 'react';
import {
    PieChart, Pie, Cell, Tooltip as RechartsTooltip, ResponsiveContainer, Legend
} from 'recharts';
import {
    Calendar, Settings, Activity, Clock, Target, Download, RefreshCw, X, ArrowLeft
} from 'lucide-react';
import { API_BASE_URL } from "../config";

const LifecycleDashboard = ({ isOpen, onClose }) => {
    const [datesList, setDatesList] = useState([]);
    const [selectedDate, setSelectedDate] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [statusMsg, setStatusMsg] = useState('等待连接 API...');

    const [uiData, setUiData] = useState({
        error: [], warning: [], normal: [], ignore: [],
        pies: { primary: [], secondary: [], truss: [], pallet: [], total: [] },
        kpi: { total: 0, exceptions: 0 }
    });

    const [config, setConfig] = useState({ log_root: '', nesting_root: '' });
    const [showConfig, setShowConfig] = useState(false);

    const [selectedPart, setSelectedPart] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);

    const [pieModalConfig, setPieModalConfig] = useState({ isOpen: false, title: '', parts: [] });

    useEffect(() => {
        if (isOpen) {
            fetchConfig();
            fetchDates();
        }
    }, [isOpen]);

    const fetchConfig = async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/api/log_config`);
            const data = await res.json();
            setConfig(data);
        } catch (e) { console.error("API 未连接"); }
    };

    const fetchDates = async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/api/log_dates`);
            const data = await res.json();
            setDatesList(data);
            if (data.length > 0) {
                setSelectedDate(data[0]);
            } else {
                setStatusMsg("后端已连接！请点击左下角配置路径。");
            }
        } catch (e) { setStatusMsg("无法连接到后端服务"); }
    };

    const saveConfig = async () => {
        try {
            await fetch(`${API_BASE_URL}/api/log_config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            setShowConfig(false);
            fetchDates();
        } catch (e) { alert("保存配置失败，请确保后端已启动。"); }
    };

    const fetchDailyData = async (dateStr, forceRefresh = false) => {
        if (!dateStr) return;
        setIsLoading(true);
        setStatusMsg(forceRefresh ? `强制重新解析 ${dateStr} 日志...` : `正在获取 ${dateStr} 数据...`);

        try {
            const response = await fetch(`${API_BASE_URL}/api/log_analyze?date_folder=${dateStr}&refresh=${forceRefresh}`);
            if (!response.ok) throw new Error(await response.text());
            const data = await response.json();
            setUiData(data);
            setStatusMsg(`${dateStr} 数据已就绪，共处理 ${data.kpi.total} 件`);
        } catch (error) {
            console.error(error);
            setStatusMsg(`分析失败: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        if (selectedDate) fetchDailyData(selectedDate, false);
    }, [selectedDate]);

    const handleExport = async () => {
        if (!selectedDate) return;
        setIsLoading(true);
        setStatusMsg("正在由后端打包带图 Excel，请稍候...");
        try {
            const response = await fetch(`${API_BASE_URL}/api/log_export?date_folder=${selectedDate}`);
            if (!response.ok) throw new Error("导出失败");

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `产线数字孪生报表_${selectedDate}.xlsx`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);

            setStatusMsg("Excel 报表已成功下载！");
        } catch (error) {
            setStatusMsg("报表下载失败，请检查后端运行状态。");
        } finally {
            setIsLoading(false);
        }
    };

    const handlePieClick = (stepKey, stepTitle, entryData) => {
        const statusName = entryData.name || entryData.payload?.name;
        const allParts = [...uiData.error, ...uiData.warning, ...uiData.normal, ...uiData.ignore];
        const filtered = allParts.filter(p => p.steps && p.steps[stepKey] === statusName);

        setPieModalConfig({
            isOpen: true,
            title: `${stepTitle} - ${statusName} (${filtered.length}件)`,
            parts: filtered
        });
    };

    const COLORS = { '正常': '#22c55e', '警戒': '#f59e0b', '异常': '#ef4444' };

    const renderPartItem = (item, colorClass) => (
        <div
            key={item.uid}
            onClick={() => { setSelectedPart(item); setIsModalOpen(true); }}
            className="cursor-pointer flex justify-between items-center p-3 mb-3 bg-white border border-slate-200 rounded-xl hover:bg-slate-50 transition-colors shadow-sm"
        >
            <div className="flex-1">
                <div className={`font-bold text-sm ${colorClass}`}>{item.part_no}</div>
                <div className="text-xs text-slate-400 font-mono mt-1">UID: {item.uid}</div>
                <div className="text-xs text-slate-500 mt-1 flex items-center">
                    <Clock size={12} className="mr-1" /> {item.duration}m | {item.status}
                </div>
            </div>
            <div className="w-20 h-20 bg-slate-50 rounded-lg flex items-center justify-center border border-slate-100 ml-2 overflow-hidden shrink-0">
                {item.img_url ? (
                    <img src={`${API_BASE_URL}${item.img_url}`} alt="CAD" className="max-w-full max-h-full object-contain" />
                ) : (
                    <span className="text-[10px] text-slate-300">无图形</span>
                )}
            </div>
        </div>
    );

    const renderPie = (data, title, stepKey, isLarge = false) => (
        <div className="flex flex-col items-center justify-center w-full h-full min-h-[160px] bg-white rounded-lg border border-slate-200 p-3 relative shadow-sm">
            <h4 className={`font-bold mb-1 absolute top-3 left-4 ${isLarge ? 'text-sm text-blue-600' : 'text-[10px] text-slate-400'}`}>
                {title}
            </h4>
            {data && data.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                            data={data}
                            innerRadius={isLarge ? "55%" : "50%"}
                            outerRadius={isLarge ? "80%" : "75%"}
                            paddingAngle={2} dataKey="value"
                            onClick={(entry) => handlePieClick(stepKey, title, entry)}
                            className="cursor-pointer"
                        >
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[entry.name]} stroke="rgba(0,0,0,0)" style={{ outline: 'none' }} />
                            ))}
                        </Pie>
                        <RechartsTooltip />
                        <Legend verticalAlign="bottom" height={isLarge ? 30 : 20} iconType="circle" wrapperStyle={{ fontSize: isLarge ? '12px' : '10px' }} />
                    </PieChart>
                </ResponsiveContainer>
            ) : (
                <div className="text-xs text-slate-300 font-mono flex-1 flex items-center justify-center">无数据</div>
            )}
        </div>
    );

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex flex-col bg-slate-50 text-slate-800 overflow-hidden font-sans animate-in fade-in duration-200">

            {/* 顶部导航栏 */}
            <header className="py-3 px-6 flex items-center justify-between border-b border-slate-200 bg-white shrink-0 shadow-sm">
                <div className="flex items-center flex-1">
                    <button onClick={onClose} className="p-2 mr-3 hover:bg-slate-100 rounded-full text-slate-500" title="返回首页">
                        <ArrowLeft size={20} />
                    </button>
                    <div className="flex flex-col justify-center">
                        <div className="text-lg font-bold text-slate-800">生产监测看板</div>
                        <div className="flex items-center mt-0.5">
                            <div className="w-2 h-2 bg-blue-500 mr-2 rounded-full"></div>
                            <p className="text-xs text-slate-500 font-mono flex items-center">
                                {statusMsg}
                                {isLoading && <div className="ml-2 w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>}
                            </p>
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-3 shrink-0">
                    <button
                        onClick={() => fetchDailyData(selectedDate, true)}
                        disabled={!selectedDate || isLoading}
                        className={`flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-all border
                            ${(!selectedDate || isLoading)
                                ? 'bg-slate-100 text-slate-400 border-slate-200 cursor-not-allowed'
                                : 'bg-white hover:bg-slate-50 text-slate-600 border-slate-200 hover:border-slate-300'}`}
                    >
                        <RefreshCw size={14} className="mr-2" />
                        获取最新实况
                    </button>

                    <button
                        onClick={handleExport}
                        disabled={!selectedDate || isLoading}
                        className={`flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-all shadow-sm
                            ${(!selectedDate || isLoading)
                                ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                                : 'bg-blue-600 hover:bg-blue-500 text-white'}`}
                    >
                        <Download size={14} className="mr-2" />
                        下载报表
                    </button>

                    <button onClick={onClose} className="p-2 hover:bg-slate-100 rounded-full text-slate-400 hover:text-slate-600 transition-colors">
                        <X size={20} />
                    </button>
                </div>
            </header>

            {/* 主体工作区 */}
            <div className="flex flex-1 overflow-hidden">

                {/* 左侧边栏 */}
                <aside className="w-64 bg-white border-r border-slate-200 flex flex-col shrink-0">
                    <div className="flex-1 py-4 overflow-y-auto">
                        <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 px-4 flex items-center">
                            <Calendar size={12} className="mr-2" /> 历史批次
                        </h2>
                        <div className="space-y-1 px-3">
                            {datesList.length > 0 ? datesList.map(date => (
                                <button
                                    key={date}
                                    onClick={() => setSelectedDate(date)}
                                    disabled={isLoading}
                                    className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all
                                        ${selectedDate === date
                                            ? 'bg-blue-50 text-blue-600 font-medium border border-blue-200'
                                            : 'text-slate-500 hover:bg-slate-50 hover:text-slate-700 border border-transparent'}`}
                                >
                                    <span className="font-mono">{date}</span>
                                </button>
                            )) : (
                                <div className="text-xs text-slate-400 text-center py-4 bg-slate-50 rounded mx-2 border border-dashed border-slate-200">无批次数据</div>
                            )}
                        </div>
                    </div>

                    <div className="p-3 border-t border-slate-200">
                        <button
                            onClick={() => setShowConfig(!showConfig)}
                            className="w-full flex items-center justify-center px-4 py-2.5 rounded-lg text-sm font-medium bg-slate-50 border border-slate-200 text-slate-600 hover:bg-slate-100 transition-colors"
                        >
                            <Settings size={14} className="mr-2" /> 引擎配置
                        </button>
                    </div>
                </aside>

                {/* 右侧主内容区 */}
                <main className="flex-1 flex flex-col relative overflow-hidden bg-slate-50">
                    {showConfig && (
                        <div className="absolute top-0 left-0 w-full bg-white/95 backdrop-blur-md border-b border-slate-200 p-6 z-50 shadow-lg">
                            <h3 className="text-base font-bold mb-4 flex items-center text-slate-700">
                                <Settings className="mr-2 text-blue-500" /> 后端路径映射配置
                            </h3>
                            <div className="space-y-4 max-w-3xl">
                                <div>
                                    <label className="block text-xs font-bold text-slate-400 mb-1.5 uppercase tracking-wider">Log 根目录绝对路径</label>
                                    <input type="text" value={config.log_root} onChange={e => setConfig({ ...config, log_root: e.target.value })}
                                        className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-sm text-slate-700 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200 transition-colors"
                                        placeholder="例如: /data/logs" />
                                </div>
                                <div>
                                    <label className="block text-xs font-bold text-slate-400 mb-1.5 uppercase tracking-wider">套料图物理库路径</label>
                                    <input type="text" value={config.nesting_root} onChange={e => setConfig({ ...config, nesting_root: e.target.value })}
                                        className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-sm text-slate-700 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200 transition-colors"
                                        placeholder="例如: /data/logs/VISUALNESTING" />
                                </div>
                                <div className="flex gap-3 pt-2">
                                    <button onClick={saveConfig} className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg text-sm font-medium text-white transition-colors">保存配置</button>
                                    <button onClick={() => setShowConfig(false)} className="bg-slate-100 hover:bg-slate-200 border border-slate-200 px-6 py-2 rounded-lg text-sm text-slate-600 transition-colors">取消</button>
                                </div>
                            </div>
                        </div>
                    )}

                    <div className={`flex-1 overflow-hidden flex transition-opacity duration-300 ${isLoading ? 'opacity-30' : 'opacity-100'}`}>
                        {/* 左：零件列表 */}
                        <div className="w-[380px] border-r border-slate-200 flex flex-col bg-white">
                            <div className="p-4 border-b border-slate-200">
                                <h3 className="font-bold text-sm text-slate-700 flex items-center">
                                    <Target size={14} className="mr-2 text-blue-500" /> 实体映射清单
                                </h3>
                            </div>
                            <div className="flex-1 overflow-y-auto p-4">
                                {uiData.error.map(item => renderPartItem(item, 'text-red-500'))}
                                {uiData.warning.map(item => renderPartItem(item, 'text-amber-500'))}
                                {uiData.normal.map(item => renderPartItem(item, 'text-green-500'))}
                            </div>
                        </div>

                        {/* 右：KPI + 饼图 */}
                        <div className="flex-1 p-6 flex flex-col bg-slate-50">
                            <div className="grid grid-cols-2 gap-4 mb-6 shrink-0">
                                <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
                                    <p className="text-xs text-slate-400 font-medium mb-1 uppercase tracking-wider">处理总零件数</p>
                                    <h3 className="text-3xl font-bold text-slate-800 font-mono">{uiData.kpi.total}</h3>
                                </div>
                                <div className={`bg-white border rounded-xl p-5 shadow-sm ${uiData.kpi.exceptions > 0 ? 'border-red-200 bg-red-50/50' : 'border-slate-200'}`}>
                                    <p className="text-xs text-slate-400 font-medium mb-1 uppercase tracking-wider">异常拦截数</p>
                                    <h3 className={`text-3xl font-bold font-mono ${uiData.kpi.exceptions > 0 ? 'text-red-500' : 'text-slate-800'}`}>{uiData.kpi.exceptions}</h3>
                                </div>
                            </div>

                            <div className="flex-1 bg-white border border-slate-200 rounded-xl p-5 flex flex-col overflow-hidden shadow-sm">
                                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center shrink-0">
                                    <Activity size={14} className="mr-2 text-green-500" /> 全局耗时分布矩阵
                                </h3>

                                <div className="flex-1 flex gap-4 px-2 pb-2 h-full">
                                    <div className="w-1/3 h-full">
                                        {renderPie(uiData.pies?.total, "总体聚合大盘", "total", true)}
                                    </div>

                                    <div className="w-2/3 grid grid-cols-2 gap-3 h-full">
                                        {renderPie(uiData.pies?.primary, "一次分拣", "primary")}
                                        {renderPie(uiData.pies?.secondary, "二次分拣", "secondary")}
                                        {renderPie(uiData.pies?.truss, "桁架分拣", "truss")}
                                        {renderPie(uiData.pies?.pallet, "码盘调度", "pallet")}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* 饼图下钻弹窗 */}
                    {pieModalConfig.isOpen && (
                        <div className="absolute inset-0 z-50 flex items-center justify-center bg-slate-900/30 backdrop-blur-sm">
                            <div className="bg-white border border-slate-200 rounded-2xl w-[600px] max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
                                <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50">
                                    <h3 className="text-base font-bold text-slate-800 flex items-center">
                                        <Target size={18} className="mr-2 text-blue-500" />
                                        {pieModalConfig.title}
                                    </h3>
                                    <button
                                        onClick={() => setPieModalConfig({ ...pieModalConfig, isOpen: false })}
                                        className="p-1.5 bg-slate-100 rounded-lg text-slate-400 hover:text-white hover:bg-red-500 transition-colors"
                                    >
                                        <X size={18} />
                                    </button>
                                </div>
                                <div className="p-4 overflow-y-auto flex-1">
                                    {pieModalConfig.parts.length > 0 ? (
                                        pieModalConfig.parts.map(item => {
                                            const statusColor = item.steps?.total === '异常' ? 'text-red-500' : (item.steps?.total === '警戒' ? 'text-amber-500' : 'text-green-500');
                                            return renderPartItem(item, statusColor);
                                        })
                                    ) : (
                                        <div className="text-center text-slate-400 py-10 text-sm">暂无符合条件的零件</div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* 零件详情弹窗 */}
                    {isModalOpen && selectedPart && (
                        <div className="absolute inset-0 z-[60] flex items-center justify-center bg-slate-900/30 backdrop-blur-sm">
                            <div className="bg-white border border-slate-200 rounded-2xl w-[500px] max-h-[80vh] flex flex-col shadow-2xl overflow-hidden">
                                <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50">
                                    <div>
                                        <h3 className="text-base font-bold text-slate-800">{selectedPart.part_no}</h3>
                                        <p className="text-xs text-slate-400 font-mono mt-0.5">UID: {selectedPart.uid} | 耗时: {selectedPart.duration}m</p>
                                    </div>
                                    <button
                                        onClick={() => setIsModalOpen(false)}
                                        className="p-1.5 bg-slate-100 rounded-lg text-slate-400 hover:text-white hover:bg-red-500 transition-colors"
                                    >
                                        <X size={18} />
                                    </button>
                                </div>
                                <div className="p-4 overflow-y-auto flex-1">
                                    {selectedPart.history && selectedPart.history.length > 0 ? (
                                        <div className="relative border-l-2 border-slate-200 ml-3 space-y-5">
                                            {selectedPart.history.map((log, idx) => {
                                                const match = log.match(/\[(.*?)\] (.*)/);
                                                const time = match ? match[1] : '';
                                                const action = match ? match[2] : log;
                                                const isError = action.includes('异常') || action.includes('超时');

                                                return (
                                                    <div key={idx} className="relative pl-5">
                                                        <div className={`absolute -left-[7px] top-1 w-3 h-3 rounded-full border-2 border-white ${isError ? 'bg-red-500' : 'bg-blue-500'}`}></div>
                                                        <div className={`text-sm font-medium ${isError ? 'text-red-500' : 'text-slate-700'}`}>{action}</div>
                                                        <div className="text-xs text-slate-400 font-mono mt-0.5">{time}</div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    ) : (
                                        <div className="text-center text-slate-400 py-10 text-sm">暂无流程记录数据</div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
};

export default LifecycleDashboard;
