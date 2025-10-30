/**
 * LCARS + Hitchhiker's Guide Node Enhancement System
 * Provides futuristic interactions and visual effects for ComfyUI nodes
 */

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Node color schemes based on type
const NODE_THEMES = {
    qwen: {
        color: "#cc99cc",
        glow: "rgba(204, 153, 204, 0.6)",
        title: "#9966cc"
    },
    wan: {
        color: "#9999ff",
        glow: "rgba(153, 153, 255, 0.6)",
        title: "#6666ff"
    },
    procedural: {
        color: "#ffcc99",
        glow: "rgba(255, 204, 153, 0.6)",
        title: "#ff9966"
    },
    default: {
        color: "#ff9900",
        glow: "rgba(255, 153, 0, 0.6)",
        title: "#ff6600"
    }
};

// Enhanced node styling
function applyLCARSStyle(node, nodeType = "default") {
    const theme = NODE_THEMES[nodeType] || NODE_THEMES.default;

    // Apply custom properties
    node.bgcolor = "#1a1a2e";
    node.color = theme.color;
    node.shape = "round";
    node.title_color = theme.title;

    // Store theme for later use
    node.lcarsTheme = theme;

    // Add custom rendering
    const originalOnDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function(ctx) {
        if (originalOnDrawForeground) {
            originalOnDrawForeground.apply(this, arguments);
        }

        // Draw LCARS corner accents
        drawLCARSAccents(ctx, this, theme);

        // Draw glow effect
        if (this.flags?.collapsed) return;
        drawGlowEffect(ctx, this, theme);
    };

    return node;
}

// Draw LCARS corner decorations
function drawLCARSAccents(ctx, node, theme) {
    const cornerSize = 20;
    const lineWidth = 4;

    ctx.strokeStyle = theme.color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";

    // Top-left corner
    ctx.beginPath();
    ctx.arc(node.size[0] - cornerSize, cornerSize, cornerSize, -Math.PI/2, 0);
    ctx.stroke();

    // Bottom-right corner
    ctx.beginPath();
    ctx.arc(cornerSize, node.size[1] - cornerSize, cornerSize, Math.PI/2, Math.PI);
    ctx.stroke();
}

// Draw glow effect around node
function drawGlowEffect(ctx, node, theme) {
    const glowSize = 8;

    // Create gradient for glow
    const gradient = ctx.createRadialGradient(
        node.size[0] / 2, node.size[1] / 2, 0,
        node.size[0] / 2, node.size[1] / 2, Math.max(node.size[0], node.size[1])
    );

    gradient.addColorStop(0, "transparent");
    gradient.addColorStop(0.8, "transparent");
    gradient.addColorStop(1, theme.glow);

    ctx.shadowColor = theme.glow;
    ctx.shadowBlur = glowSize;
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 2;

    // Draw glow outline
    const margin = 5;
    ctx.strokeRect(-margin, -margin, node.size[0] + margin * 2, node.size[1] + margin * 2);

    // Reset shadow
    ctx.shadowBlur = 0;
}

// Add status indicator widget
function addStatusIndicator(node, defaultStatus = "active") {
    const statusWidget = node.addWidget("button", "status", defaultStatus, () => {
        // Cycle through statuses
        const statuses = ["active", "processing", "error"];
        const currentIndex = statuses.indexOf(statusWidget.value);
        statusWidget.value = statuses[(currentIndex + 1) % statuses.length];
    });

    statusWidget.serialize = false; // Don't include in workflow

    // Custom draw for status widget
    const originalDraw = statusWidget.draw;
    statusWidget.draw = function(ctx, node, width, y, height) {
        const status = this.value;
        const colors = {
            active: "#00ff00",
            processing: "#ff9900",
            error: "#ff3333"
        };

        // Draw status indicator circle
        ctx.fillStyle = colors[status] || colors.active;
        ctx.beginPath();
        ctx.arc(width - 20, y + height/2, 6, 0, Math.PI * 2);
        ctx.fill();

        // Draw status text
        ctx.fillStyle = "#e0e0e0";
        ctx.font = "10px Orbitron, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(status.toUpperCase(), 10, y + height/2 + 4);
    };

    return statusWidget;
}

// Enhanced number widget with LCARS slider
function createLCARSSlider(node, name, value, callback, options = {}) {
    const min = options.min ?? 0;
    const max = options.max ?? 1;
    const step = options.step ?? 0.01;

    const widget = {
        type: "number",
        name: name,
        value: value,
        options: { min, max, step },
        callback: callback,

        draw: function(ctx, node, width, y, height) {
            // Draw label
            ctx.fillStyle = "#e0e0e0";
            ctx.font = "11px Orbitron, sans-serif";
            ctx.textAlign = "left";
            ctx.fillText(this.name, 10, y + 12);

            // Draw value
            ctx.textAlign = "right";
            ctx.fillText(this.value.toFixed(2), width - 10, y + 12);

            // Draw slider track
            const trackY = y + height - 8;
            const trackWidth = width - 20;
            const trackX = 10;

            // Background track
            ctx.fillStyle = "rgba(255, 153, 0, 0.2)";
            ctx.roundRect(trackX, trackY, trackWidth, 6, 3);
            ctx.fill();

            // Filled track
            const fillWidth = ((this.value - min) / (max - min)) * trackWidth;
            const gradient = ctx.createLinearGradient(trackX, trackY, trackX + trackWidth, trackY);
            gradient.addColorStop(0, "#cc6600");
            gradient.addColorStop(1, "#ff9900");

            ctx.fillStyle = gradient;
            ctx.roundRect(trackX, trackY, fillWidth, 6, 3);
            ctx.fill();

            // Thumb
            const thumbX = trackX + fillWidth;
            ctx.fillStyle = "#ff9900";
            ctx.shadowColor = "rgba(255, 153, 0, 0.6)";
            ctx.shadowBlur = 10;
            ctx.beginPath();
            ctx.arc(thumbX, trackY + 3, 8, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
        },

        mouse: function(event, pos, node) {
            if (event.type === "pointermove" || event.type === "pointerdown") {
                const localY = pos[1] - this.last_y;
                if (localY > 20 && localY < this.height) {
                    const trackWidth = node.size[0] - 20;
                    const trackX = 10;
                    const normalizedX = Math.max(0, Math.min(1, (pos[0] - trackX) / trackWidth));
                    const newValue = min + normalizedX * (max - min);

                    // Round to step
                    this.value = Math.round(newValue / step) * step;

                    if (this.callback) {
                        this.callback(this.value, node, this);
                    }

                    return true;
                }
            }
            return false;
        }
    };

    node.addCustomWidget(widget);
    return widget;
}

// Progress bar widget
function addProgressBar(node, name = "progress") {
    const widget = {
        type: "progress",
        name: name,
        value: 0,
        serialize: false,

        draw: function(ctx, node, width, y, height) {
            // Draw label
            ctx.fillStyle = "#e0e0e0";
            ctx.font = "10px Orbitron, sans-serif";
            ctx.textAlign = "left";
            ctx.fillText(this.name.toUpperCase(), 10, y + 12);

            // Draw percentage
            ctx.textAlign = "right";
            ctx.fillText(`${Math.round(this.value * 100)}%`, width - 10, y + 12);

            // Draw progress bar
            const barY = y + height - 10;
            const barWidth = width - 20;
            const barX = 10;
            const barHeight = 8;

            // Background
            ctx.fillStyle = "rgba(10, 10, 10, 0.8)";
            ctx.roundRect(barX, barY, barWidth, barHeight, 4);
            ctx.fill();

            // Progress fill
            const fillWidth = barWidth * this.value;
            const gradient = ctx.createLinearGradient(barX, barY, barX + barWidth, barY);
            gradient.addColorStop(0, "#ff9900");
            gradient.addColorStop(0.5, "#cc6600");
            gradient.addColorStop(1, "#ff9900");

            ctx.fillStyle = gradient;
            ctx.shadowColor = "rgba(255, 153, 0, 0.6)";
            ctx.shadowBlur = 10;
            ctx.roundRect(barX, barY, fillWidth, barHeight, 4);
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    };

    node.addCustomWidget(widget);
    return widget;
}

// Register extension with ComfyUI
app.registerExtension({
    name: "Studio42.LCARSTheme",

    async setup() {
        // Load LCARS font
        const font = new FontFace('Orbitron', 'url(https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap)');
        await font.load();
        document.fonts.add(font);
    },

    async nodeCreated(node) {
        // Determine node type from name
        let nodeType = "default";
        if (node.type.includes("Qwen")) nodeType = "qwen";
        else if (node.type.includes("Wan")) nodeType = "wan";
        else if (node.type.includes("Procedural")) nodeType = "procedural";

        // Apply LCARS styling
        applyLCARSStyle(node, nodeType);

        // Add status indicator for processor nodes
        if (node.type.includes("Qwen") || node.type.includes("Wan") || node.type.includes("Procedural")) {
            addStatusIndicator(node);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Add custom widget types
        if (nodeData.category?.includes("Studio42/LCARS")) {
            // Enhance with LCARS widgets
            const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(canvas, options) {
                if (originalGetExtraMenuOptions) {
                    originalGetExtraMenuOptions.apply(this, arguments);
                }

                options.push({
                    content: "🌟 Toggle Glow Effect",
                    callback: () => {
                        this.glowEnabled = !this.glowEnabled;
                        this.setDirtyCanvas(true);
                    }
                });
            };
        }
    }
});

// Export utilities for custom nodes
export {
    applyLCARSStyle,
    addStatusIndicator,
    createLCARSSlider,
    addProgressBar,
    NODE_THEMES
};
