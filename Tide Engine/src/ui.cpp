#include "ui.h"
#include "vk_engine.h"
#include "settings.h"
#include "camera.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

void Ui::init(VulkanEngine& eng, VkFormat colorFormat,
              uint32_t minImageCount, uint32_t imageCount) {
    m_colorFormat = colorFormat;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // Darker, slightly rounded theme.
    ImGui::StyleColorsDark();
    ImGuiStyle& st = ImGui::GetStyle();
    st.WindowRounding = 6.0f;
    st.FrameRounding  = 4.0f;
    st.GrabRounding   = 4.0f;
    st.WindowPadding  = ImVec2(10, 10);
    ImVec4* c = st.Colors;
    c[ImGuiCol_WindowBg]        = ImVec4(0.01f, 0.01f, 0.015f, 0.78f);
    c[ImGuiCol_TitleBg]         = ImVec4(0.02f, 0.02f, 0.03f, 1.00f);
    c[ImGuiCol_TitleBgActive]   = ImVec4(0.06f, 0.06f, 0.08f, 1.00f);
    c[ImGuiCol_FrameBg]         = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
    c[ImGuiCol_FrameBgHovered]  = ImVec4(0.16f, 0.16f, 0.20f, 1.00f);
    c[ImGuiCol_FrameBgActive]   = ImVec4(0.20f, 0.20f, 0.26f, 1.00f);
    c[ImGuiCol_SliderGrab]      = ImVec4(0.35f, 0.55f, 0.85f, 1.00f);
    c[ImGuiCol_SliderGrabActive]= ImVec4(0.45f, 0.65f, 0.95f, 1.00f);
    c[ImGuiCol_Header]          = ImVec4(0.14f, 0.14f, 0.18f, 1.00f);

    ImGui_ImplGlfw_InitForVulkan(eng.window(), true);

    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_3;
    info.Instance = eng.instance();
    info.PhysicalDevice = eng.physicalDevice();
    info.Device = eng.device();
    info.QueueFamily = eng.queueFamily();
    info.Queue = eng.queue();
    info.MinImageCount = minImageCount;
    info.ImageCount = imageCount;
    info.DescriptorPoolSize = 64; // backend creates its own pool
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &m_colorFormat;

    ImGui_ImplVulkan_Init(&info);
}

void Ui::beginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Ui::buildPanel(Settings& s, Camera& cam, bool camPlaying, bool& playToggled,
                    float dt, float cpuMs) {
    // Real frame time (dt) drives the FPS + graph; CPU work time (cpuMs) is a
    // separate readout. Refresh displayed numbers once per real second.
    m_acc += dt;
    m_frames++;
    m_realMax = std::max(m_realMax, dt);
    m_cpuAcc += cpuMs;
    if (m_acc >= 1.0f) {
        m_dispAvgMs = (m_frames > 0) ? (m_acc / m_frames) * 1000.0f : 0.0f;
        m_dispMaxMs = m_realMax * 1000.0f;
        m_dispCpuMs = (m_frames > 0) ? (m_cpuAcc / m_frames) : 0.0f;
        m_acc = 0.0f;
        m_frames = 0;
        m_realMax = 0.0f;
        m_cpuAcc = 0.0f;
    }
    float avgFps = m_dispAvgMs > 0.0f ? 1000.0f / m_dispAvgMs : 0.0f;
    float maxFps = m_dispMaxMs > 0.0f ? 1000.0f / m_dispMaxMs : 0.0f;

    // Plot the real frame time (actual frame rate) at a fixed sample rate.
    int hz = s.frameGraphHz > 0 ? s.frameGraphHz : 1;
    float sampleInterval = 1.0f / (float)hz;
    m_plotAcc += dt;
    m_plotFrames++;
    if (m_plotAcc >= sampleInterval) {
        m_msHistory[m_msHead] = (m_plotFrames > 0) ? (m_plotAcc / m_plotFrames) * 1000.0f : 0.0f;
        m_msHead = (m_msHead + 1) % kHistory;
        m_plotAcc = 0.0f;
        m_plotFrames = 0;
    }

    // Pin the panel to the top-right corner; fixed size, not movable/resizable.
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x - 10.0f,
                                   vp->WorkPos.y + 10.0f),
                            ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 800.0f), ImGuiCond_Always);

    ImGui::Begin("Tide Engine", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    ImGui::Text("%.2f ms  (%.0f FPS)", m_dispAvgMs, avgFps);
    ImGui::SetItemTooltip("Actual frame time / on-screen FPS (VSync-capped).");
    ImGui::Text("Max: %.2f ms  (%.0f FPS)", m_dispMaxMs, maxFps);
    ImGui::SetItemTooltip("Slowest frame in the last second (worst-case).");
    ImGui::Text("CPU: %.2f ms", m_dispCpuMs);
    ImGui::SetItemTooltip("Engine CPU work/frame (excludes VSync idle) = headroom.");

    ImGui::Checkbox("VSync", &s.vsync);
    ImGui::SetItemTooltip("On: FIFO (locked to refresh). Off: IMMEDIATE (uncapped, may tear).");
    ImGui::Checkbox("Frame graph", &s.showFrameGraph);
    ImGui::SetNextItemWidth(150.0f);
    ImGui::SliderInt("Graph Hz", &s.frameGraphHz, 1, 60);
    ImGui::SetItemTooltip("Graph samples per second.");
    ImGui::SetNextItemWidth(150.0f);
    ImGui::SliderFloat("Graph Max", &s.frameGraphMaxMs, 2.0f, 200.0f, "%.0f ms");
    ImGui::SetItemTooltip("Fixed Y-axis top of the frame-time graph.");

    ImGui::SeparatorText("Sun");
    ImGui::PushItemWidth(150.0f); // fixed slider width
    ImGui::Checkbox("Animate", &s.sunAnimate);
    if (s.sunAnimate) {
        // Min/max bounds shown side by side; azimuth and elevation each on one row.
        ImGui::PushItemWidth(70.0f);
        ImGui::SliderFloat("##azMin", &s.sunAzimuthMin, 0.0f, 360.0f, "%.0f");
        ImGui::SameLine();
        ImGui::SliderFloat("Azimuth##range", &s.sunAzimuthMax, 0.0f, 360.0f, "%.0f");
        ImGui::SliderFloat("##elMin", &s.sunElevationMin, -10.0f, 90.0f, "%.0f");
        ImGui::SameLine();
        ImGui::SliderFloat("Elevation##range", &s.sunElevationMax, -10.0f, 90.0f, "%.0f");
        ImGui::PopItemWidth();
        ImGui::SliderFloat("Speed", &s.sunAnimSpeed, 0.02f, 1.0f, "%.2f");
        ImGui::Text("Now: az %.0f  el %.0f", s.sunAzimuthDeg, s.sunElevationDeg);
    } else {
        ImGui::SliderFloat("Azimuth",   &s.sunAzimuthDeg,   0.0f, 360.0f);
        ImGui::SliderFloat("Elevation", &s.sunElevationDeg, -10.0f, 90.0f);
    }
    ImGui::SliderFloat("Intensity", &s.sunIntensity,    0.0f, 20.0f);
    ImGui::SliderFloat("Ambient",   &s.ambient,         0.0f, 1.0f);

    ImGui::SeparatorText("Shadows (RT)");
    ImGui::Checkbox("Enabled", &s.shadowsEnabled);
    ImGui::SliderFloat("Softness", &s.sunAngularSize, 0.0f, 5.0f, "%.2f deg");
    ImGui::SliderInt("Samples",    &s.shadowSamples,  1, 16);

    // ---- Denoiser / AA: one selector; only the active mode's params are shown. ----
    ImGui::SeparatorText("Denoiser / AA");
    ImGui::TextDisabled("(?)");
    ImGui::SetItemTooltip("Temporal/SVGF denoise shadows and AO only. They do not cover\n"
                          "reflections; GI has its own heuristic denoise approach.\n"
                          "DLSS RR is the general AA + denoise solution for shadows, AO,\n"
                          "and reflections.");
    const char* denoisers[] = {"Off", "Temporal", "SVGF", "DLSS Ray Reconstruction"};
    ImGui::Combo("Mode##denoiser", &s.denoiser, denoisers, IM_ARRAYSIZE(denoisers));
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Off: raw noisy shadow.\n"
                          "Temporal: reproject + EMA (time only).\n"
                          "SVGF: temporal + edge-aware a-trous (time + space).\n"
                          "DLSS RR: AI denoise + upscale (NGX, render at lower res).\n");
    // Map the single selector to the engine-facing fields.
    s.dlssEnabled       = (s.denoiser == 3);
    s.shadowDenoiseMode = (s.denoiser <= 2) ? s.denoiser : 0;

    if (s.denoiser == 1) {            // Temporal
        ImGui::SliderFloat("History", &s.shadowHistAlpha, 0.001f, 0.5f, "%.3f");
    } else if (s.denoiser == 2) {     // SVGF
        ImGui::SliderFloat("History", &s.shadowHistAlpha, 0.001f, 0.5f, "%.3f");
        ImGui::SliderInt("A-trous iters",   &s.svgfIterations,  1, 5);
        ImGui::SliderFloat("Phi normal",    &s.svgfPhiNormal,   1.0f, 128.0f, "%.0f");
        ImGui::SliderFloat("Phi depth",     &s.svgfPhiDepth,    0.05f, 4.0f, "%.2f");
    } else if (s.denoiser == 3) {     // DLSS Ray Reconstruction
        const char* dlssModes[] = {"Performance", "Balanced", "Quality", "Ultra Performance", "DLAA"};
        ImGui::Combo("Quality", &s.dlssQuality, dlssModes, IM_ARRAYSIZE(dlssModes));
        if (!s.dlssAvailable)
            ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "Status: NOT AVAILABLE (native)");
        else if (s.dlssActive)
            ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1), "Status: ACTIVE  %ux%u -> %ux%u",
                               s.renderW, s.renderH, s.displayW, s.displayH);
        else
            ImGui::TextColored(ImVec4(1, 1, 0.4f, 1), "Status: off (native %ux%u)", s.displayW, s.displayH);
    }
    ImGui::Checkbox("Debug: motion vectors", &s.debugMotionVecs);

    ImGui::SeparatorText("Ambient Occlusion (RTAO)");
    ImGui::Checkbox("AO enabled", &s.aoEnabled);
    ImGui::SetItemTooltip("Ray-traced AO against the scene TLAS. Darkens the ambient/indirect\n"
                          "term in corners and contact points. Off = ambient unmodified.\n"
                          "Shares the shadow denoiser (Temporal/SVGF/RR), so few rays suffice.");
    if (s.aoEnabled) {
        ImGui::SliderInt("AO rays",      &s.aoSamples,   1, 16);
        ImGui::SetItemTooltip("Cosine-weighted hemisphere rays per pixel. Denoised downstream,\n"
                              "so 2-4 is usually enough.");
        ImGui::SliderFloat("AO radius",  &s.aoRadius,    0.05f, 3.0f, "%.2f");
        ImGui::SetItemTooltip("World-space max occluder distance. Small = tight contact shadows,\n"
                              "large = broad ambient darkening.");
        ImGui::SliderFloat("AO intensity", &s.aoIntensity, 0.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("AO bias",    &s.aoBias,      0.0f, 0.2f, "%.3f");
        ImGui::SetItemTooltip("Ray origin offset along the normal (fixes self-occlusion acne).");
    }

    ImGui::SeparatorText("Reflections (SSR)");
    const char* reflModes[] = {"Off", "SSR", "SSR + RT", "RT only"};
    ImGui::Combo("Mode##refl", &s.reflectionsMode, reflModes, IM_ARRAYSIZE(reflModes));
    ImGui::SetItemTooltip("Reflections on opaque surfaces. SSR reflects the lit scene visible on\n"
                          "screen; SSR + RT traces the scene TLAS where the screen-space ray misses;\n"
                          "RT only skips the screen-space march and always traces. Transparents do not reflect.");
    if (s.reflectionsMode > 0) {
        // SSR march params only matter when the screen-space march runs (SSR / SSR+RT,
        // not RT-only). Grey them out otherwise so they read as inactive.
        bool ssrActive = (s.reflectionsMode == 1 || s.reflectionsMode == 2);
        ImGui::BeginDisabled(!ssrActive);
        ImGui::SliderInt("SSR steps",        &s.ssrSteps,        8, 128);
        ImGui::SetItemTooltip("Screen-space march samples. More = catches farther/grazing hits, costs more.");
        ImGui::SliderFloat("SSR distance",   &s.ssrMaxDistance,  2.0f, 100.0f, "%.0f");
        ImGui::SetItemTooltip("World-space length of the reflected ray march.");
        ImGui::SliderFloat("SSR thickness",  &s.ssrThickness,    0.05f, 2.0f, "%.2f");
        ImGui::SetItemTooltip("Assumed surface thickness (depth buffer has no back faces): a hit counts\n"
                              "only if the ray passes within this depth of the front surface.\n"
                              "Too small = reflections drop out; too large = smears behind objects.");
        ImGui::EndDisabled();
        ImGui::SliderFloat("Max roughness",  &s.ssrMaxRoughness, 0.05f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Skip reflections on surfaces rougher than this (DDGI covers diffuse).");
        ImGui::SliderFloat("Intensity##refl",&s.reflectionIntensity, 0.0f, 2.0f, "%.2f");
    }

    ImGui::SeparatorText("Volumetric Fog");
    ImGui::Checkbox("Fog enabled", &s.fogEnabled);
    ImGui::SetItemTooltip("Job: the froxel volumetric god-ray pipeline (scatter -> integrate -> apply).\n"
                          "Result: light shafts in the air. Off = scene renders exactly as before.");
    if (s.fogEnabled) {
        const char* fogQ[] = {"160x90x64", "240x135x96", "320x180x128", "480x270x160 (Ultra)"};
        ImGui::Combo("Froxel res", &s.fogQuality, fogQ, IM_ARRAYSIZE(fogQ));
        ImGui::SetItemTooltip("Job: 3D froxel grid resolution (X,Y screen-aligned, Z depth slices).\n"
                              "Result: higher XY resolves thin light shafts (small blind gaps); crisper, less blocky.\n"
                              "Higher = sharper + more realistic shafts + less leak, but more GPU cost. Ultra is heavy.");
        ImGui::Checkbox("Jitter (temporal AA)", &s.fogJitter);
        ImGui::SetItemTooltip("Job: nudges each froxel's shadow sample a little every frame; temporal averages them.\n"
                              "Result: turns hard 0/1 shadow edges into soft penumbra.\n"
                              "On = smoother IF the view is steady, but adds shimmer while moving.\n"
                              "Off = perfectly stable but blocky (use Volume blur instead).");
        ImGui::SliderInt("Blur radius", &s.fogBlurRadius, 0, 3);
        ImGui::SetItemTooltip("Job: deterministic NxNxN box blur of the froxel volume (N=2r+1).\n"
                              "Result: smooths blocky beams WITHOUT any motion/shimmer (unlike jitter).\n"
                              "Higher = smoother but softer/wider beams + more cost. 0 = off (blocky).");
        ImGui::SliderFloat("Density",     &s.fogDensity,      0.0f, 0.3f, "%.3f");
        ImGui::SetItemTooltip("Job: extinction per world unit (how thick the medium is).\n"
                              "Result: overall fog amount.\n"
                              "Higher = denser haze, brighter beams, less visibility. Lower = thinner/clearer.");
        ImGui::SliderFloat("Scatter",     &s.fogScatter,      0.0f, 2.0f, "%.2f");
        ImGui::SetItemTooltip("Job: scattering albedo (how much in-scattered sunlight a froxel emits).\n"
                              "Result: brightness of the beams.\n"
                              "Higher = brighter shafts. Lower = dimmer (density unchanged).");
        ImGui::SliderFloat("Anisotropy",  &s.fogAnisotropy,  -0.9f, 0.95f, "%.2f");
        ImGui::SetItemTooltip("Job: Henyey-Greenstein g, the scattering directionality.\n"
                              "Result: how concentrated the glow is around the sun direction.\n"
                              "Higher (->1) = sharp forward beams (looking toward sun). 0 = uniform glow. Negative = back-scatter.");
        ImGui::SliderFloat("Fog ambient", &s.fogAmbient,      0.0f, 0.2f, "%.3f");
        ImGui::SetItemTooltip("Job: constant in-scatter added to every froxel (sky fill in the medium).\n"
                              "Result: base haze even where the sun doesn't reach.\n"
                              "Higher = milky/foggy everywhere. Lower = only lit shafts show, shadows stay clear.");
        ImGui::SliderFloat("Temporal",    &s.fogTemporalAlpha, 0.001f, 1.0f, "%.3f");
        ImGui::SetItemTooltip("Job: EMA blend weight of THIS frame vs accumulated history (jitter averaging).\n"
                              "Result: how fast the volume converges vs how stable it is.\n"
                              "Lower = smoother/more stable but laggier. Higher = responsive but noisier/shimmery.");
        ImGui::SliderFloat("Max distance",&s.fogMaxDistance,  5.0f, 200.0f, "%.0f");
        ImGui::SetItemTooltip("Job: far extent of the froxel volume (world units); Z slices pack into this range.\n"
                              "Result: how far fog is computed.\n"
                              "Higher = fog reaches farther but coarser per-slice (more blocky far away). Lower = denser slices up close.");

        ImGui::Checkbox("Local fog box", &s.fogBoxEnabled);
        ImGui::SetItemTooltip("Job: confine fog density to a world-space box with a soft edge, instead\n"
                              "of filling the whole scene.\n"
                              "Result: no fog outside the box (looking out a window is clear), and froxels\n"
                              "straddling a wall fade to ~0 there -> kills the in/out interpolation leak.\n"
                              "Off = uniform global density (fog everywhere).");
        if (s.fogBoxEnabled) {
            ImGui::SliderFloat("Box edge softness", &s.fogBoxEdge, 0.0f, 3.0f, "%.2f");
            ImGui::SetItemTooltip("Smoothstep falloff width at the box faces (world units).\n"
                                  "Higher = gradual fade in/out at the box border (no hard fog wall).");
            ImGui::Checkbox("Show fog box", &s.fogDebugBox);
            ImGui::SetItemTooltip("Draws the fog box as a cyan wireframe so you can fit it to the room.");
            ImGui::Checkbox("Manual box placement", &s.fogBoxManual);
            ImGui::SetItemTooltip("Off = auto-fit the box to the scene bounds each frame.\n"
                                  "On = place it by hand. Fit it inside the room walls so fog stops at them.");
            ImGui::BeginDisabled(!s.fogBoxManual); // when auto, the engine drives these
            float center[3] = {(s.fogBoxMin.x + s.fogBoxMax.x) * 0.5f,
                               (s.fogBoxMin.y + s.fogBoxMax.y) * 0.5f,
                               (s.fogBoxMin.z + s.fogBoxMax.z) * 0.5f};
            float size[3]   = {s.fogBoxMax.x - s.fogBoxMin.x,
                               s.fogBoxMax.y - s.fogBoxMin.y,
                               s.fogBoxMax.z - s.fogBoxMin.z};
            bool ch = false;
            ch |= ImGui::DragFloat3("Box center", center, 0.05f);
            ch |= ImGui::DragFloat3("Box size (XYZ)", size, 0.05f, 0.1f, 10000.0f);
            if (ch) {
                for (int i = 0; i < 3; i++) if (size[i] < 0.1f) size[i] = 0.1f;
                s.fogBoxMin = glm::vec3(center[0] - size[0] * 0.5f, center[1] - size[1] * 0.5f,
                                        center[2] - size[2] * 0.5f);
                s.fogBoxMax = glm::vec3(center[0] + size[0] * 0.5f, center[1] + size[1] * 0.5f,
                                        center[2] + size[2] * 0.5f);
            }
            ImGui::EndDisabled();
        }
    }

    ImGui::SeparatorText("Global Illumination (DDGI)");
    ImGui::Checkbox("GI enabled", &s.giEnabled);
    ImGui::SetItemTooltip("Job: real diffuse indirect light (DDGI probe grid) replaces the flat ambient.\n"
                          "Result: sunlight bounces off floor/walls and lights shadowed areas with color.\n"
                          "Off = flat constant ambient (exactly as before).");
    if (s.giEnabled) {
        ImGui::SliderFloat("GI intensity (1=physical)", &s.giIntensity, 0.0f, 8.0f, "%.2f");
        ImGui::SetItemTooltip("Job: master multiplier on the indirect irradiance (the indirect-vs-direct ratio).\n"
                              "Result: strength of bounce lighting. 1 = physically correct; >1 = exaggerate.\n"
                              "This sets the RELATIVE level; use Exposure for absolute brightness.");
        ImGui::SliderFloat("Sky GI strength", &s.giSkyIntensity, 0.0f, 4.0f, "%.2f");
        ImGui::SetItemTooltip("Job: how strongly the sky (seen through windows on ray miss) lights the scene.\n"
                              "Result: cool fill from the window/sky. Free knob — the procedural sky isn't calibrated.");
        ImGui::SliderFloat("Multi-bounce gain", &s.giMultiBounce, 0.0f, 2.0f, "%.2f");
        ImGui::SetItemTooltip("Job: feedback of the previous frame's indirect into each probe ray's hit (extra bounces).\n"
                              "Result: the scene fills up naturally as energy accumulates. 1 = physical, 0 = single bounce only, >1 = exaggerate.");
        ImGui::SliderFloat("Hysteresis", &s.giHysteresis, 0.5f, 0.995f, "%.3f");
        ImGui::SetItemTooltip("Job: probe temporal blend (keeps this fraction of the previous frame).\n"
                              "Result: higher = stabler/less noise but slower to react to sun moves. Lower = responsive but noisier.");
        ImGui::SliderInt("Rays / probe", &s.giRaysPerProbe, 8, 128);
        ImGui::SetItemTooltip("Job: rays traced per probe each frame (hemisphere sampling).\n"
                              "Result: more = less noise/faster convergence, more GPU cost. 64 is a good default.");
        ImGui::SliderFloat("Normal bias", &s.giNormalBias, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Job: offsets the shading point along its normal before sampling probes.\n"
                              "Result: fixes self-occlusion (dark seams) at corners. Too high = light leaks.");
        // Probe grid resolution: changing any axis rebuilds the atlases (recreateSwapchain).
        int probes[3] = {s.giProbesX, s.giProbesY, s.giProbesZ};
        if (ImGui::SliderInt3("Probes XYZ", probes, 2, 32)) {
            s.giProbesX = probes[0]; s.giProbesY = probes[1]; s.giProbesZ = probes[2];
        }
        ImGui::SetItemTooltip("Job: world-space probe grid resolution (rebuilds the atlases).\n"
                              "Result: more probes = finer indirect detail + cost. The grid wraps the scene bounds.");

        ImGui::Checkbox("Probe relocation", &s.giRelocation);
        ImGui::SetItemTooltip("Job: each frame, slide probes out of walls/furniture into open space\n"
                              "(using their own ray distances).\n"
                              "Result: fixes black pockets in tight gaps (under tables) and hard lines where\n"
                              "a probe sat inside geometry. Off = probes stay fixed on the grid.");

        ImGui::Checkbox("Show probe grid", &s.giDebugProbes);
        ImGui::SetItemTooltip("Draws each probe as a small sphere colored by its irradiance.\n"
                              "Probes classified invalid (inside/behind geometry) show RED.\n"
                              "Use it to see which probes sit OUTSIDE the room (the leak sources).");
        ImGui::Checkbox("Manual grid placement", &s.giGridManual);
        ImGui::SetItemTooltip("Off = auto-fit the grid to the scene bounds each frame.\n"
                              "On = place the grid by hand. Fit it INSIDE the room walls so no probe\n"
                              "sits in the empty exterior -> kills the dominant thin-wall leak.");
        ImGui::BeginDisabled(!s.giGridManual); // when auto, the engine drives these
        // Center + per-axis size (scale) — more intuitive than two corners.
        float center[3] = {(s.giGridMin.x + s.giGridMax.x) * 0.5f,
                           (s.giGridMin.y + s.giGridMax.y) * 0.5f,
                           (s.giGridMin.z + s.giGridMax.z) * 0.5f};
        float size[3]   = {s.giGridMax.x - s.giGridMin.x,
                           s.giGridMax.y - s.giGridMin.y,
                           s.giGridMax.z - s.giGridMin.z};
        bool ch = false;
        ch |= ImGui::DragFloat3("Grid center", center, 0.05f);
        ch |= ImGui::DragFloat3("Grid size (XYZ)", size, 0.05f, 0.1f, 10000.0f);
        if (ch) {
            for (int i = 0; i < 3; i++) if (size[i] < 0.1f) size[i] = 0.1f;
            s.giGridMin = glm::vec3(center[0] - size[0] * 0.5f, center[1] - size[1] * 0.5f,
                                    center[2] - size[2] * 0.5f);
            s.giGridMax = glm::vec3(center[0] + size[0] * 0.5f, center[1] + size[1] * 0.5f,
                                    center[2] + size[2] * 0.5f);
        }
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Grid center (world pos) + per-axis size. Shrink each axis to fit the\n"
                              "grid inside the room walls; the spheres show where the probes land.");
    }

    ImGui::SeparatorText("Tonemap");
    const char* tonemappers[] = {"ACES (filmic)", "AgX"};
    ImGui::Combo("Curve", &s.tonemapper, tonemappers, IM_ARRAYSIZE(tonemappers));
    ImGui::SetItemTooltip("ACES: punchy/contrasty, but hue-shifts bright saturated colors\n"
                          "(god rays, sun, GI bounce).\n"
                          "AgX: hue-preserving, desaturates highlights to white. Cleaner\n"
                          "for the bright volumetric/GI look; slightly softer contrast.");
    ImGui::SliderFloat("Exposure",  &s.exposure,        0.1f, 200.0f);

    ImGui::SeparatorText("Bloom");
    ImGui::Checkbox("Bloom enabled", &s.bloomEnabled);
    ImGui::SetItemTooltip("Energy-conserving HDR bloom (COD/Jimenez dual-filter mip chain).\n"
                          "Bright highlights (god rays, sun, GI) bleed light into surroundings.");
    if (s.bloomEnabled) {
        ImGui::SliderFloat("Intensity##bloom", &s.bloomIntensity, 0.0f, 0.3f, "%.3f");
        ImGui::SetItemTooltip("Scene<->bloom blend (energy-conserving). ~0.04 is physical; higher = dreamier.");
        ImGui::SliderFloat("Radius##bloom", &s.bloomRadius, 0.5f, 3.0f, "%.2f");
        ImGui::SetItemTooltip("Upsample tent spread (scatter). Higher = wider, softer glow.");
        ImGui::SliderFloat("Threshold##bloom", &s.bloomThreshold, 0.0f, 5.0f, "%.2f");
        ImGui::SetItemTooltip("Soft-knee bright-pass. 0 = thresholdless (pure PBR, whole image blooms).\n"
                              "Raise to bloom only bright areas (more classic/artistic).");
        if (s.bloomThreshold > 0.0f) {
            ImGui::SliderFloat("Knee##bloom", &s.bloomKnee, 0.0f, 1.0f, "%.2f");
            ImGui::SetItemTooltip("Soft-knee width around the threshold (smoother roll-in).");
        }
    }
    ImGui::PopItemWidth();

    // ---- Camera ----
    ImGui::SeparatorText("Camera");
    ImGui::SetNextItemWidth(150.0f);
    ImGui::SliderFloat("FOV", &cam.fovDeg, 30.0f, 120.0f, "%.0f deg");
    ImGui::SetItemTooltip("Vertical field of view. Higher = wider angle (more in frame, stronger perspective).");

    // ---- Camera path: capture waypoints, reorder/delete, play a flythrough. ----
    ImGui::SeparatorText("Camera Path");
    ImGui::BeginDisabled(camPlaying); // no editing mid-playback
    if (ImGui::Button("Add") && s.camPathCount < kMaxWaypoints) {
        CamWaypoint& w = s.camPath[s.camPathCount++];
        w.pos = cam.position; w.yaw = cam.yaw; w.pitch = cam.pitch;
    }
    ImGui::SetItemTooltip("Append the current camera transform as the next waypoint.");
    ImGui::SameLine();
    if (ImGui::Button("Clear")) s.camPathCount = 0;

    if (s.camPathCount == 0) {
        ImGui::TextDisabled("(empty)");
    }
    // Each row: "Point N" with small square up/down/delete buttons aligned to its right.
    const float sq = ImGui::GetFontSize() + 2.0f; // compact square buttons
    int del = -1, moveUp = -1, moveDown = -1, setHere = -1;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(3.0f, 3.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
    for (int i = 0; i < s.camPathCount; i++) {
        ImGui::PushID(i);
        ImGui::AlignTextToFramePadding();
        ImGui::Text("Point %d", i + 1);
        ImGui::SameLine(70.0f); // align the button cluster across rows
        ImGui::BeginDisabled(i == 0);
        if (ImGui::Button("^", ImVec2(sq, sq))) moveUp = i;
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(i == s.camPathCount - 1);
        if (ImGui::Button("v", ImVec2(sq, sq))) moveDown = i;
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("o", ImVec2(sq, sq))) setHere = i;
        ImGui::SetItemTooltip("Overwrite this waypoint with the current camera transform.");
        ImGui::SameLine();
        if (ImGui::Button("x", ImVec2(sq, sq))) del = i;
        ImGui::PopID();
    }
    ImGui::PopStyleVar(2);
    if (setHere >= 0) {
        CamWaypoint& w = s.camPath[setHere];
        w.pos = cam.position; w.yaw = cam.yaw; w.pitch = cam.pitch;
    }
    if (del >= 0) {
        for (int i = del; i < s.camPathCount - 1; i++) s.camPath[i] = s.camPath[i + 1];
        s.camPathCount--;
    }
    if (moveUp > 0)   std::swap(s.camPath[moveUp], s.camPath[moveUp - 1]);
    if (moveDown >= 0 && moveDown < s.camPathCount - 1)
        std::swap(s.camPath[moveDown], s.camPath[moveDown + 1]);
    ImGui::EndDisabled(); // camPlaying — editing controls above are locked while playing

    // Speed + Loop stay live during playback so you can tune the motion in real time.
    ImGui::SetNextItemWidth(150.0f);
    ImGui::SliderFloat("Speed##campath", &s.camPathSpeed, 0.01f, 5.0f, "%.1f u/s");
    ImGui::Checkbox("Loop", &s.camPathLoop);
    ImGui::SetItemTooltip("Restart from the first waypoint at the end instead of stopping.");

    ImGui::BeginDisabled(s.camPathCount < 2);
    if (ImGui::Button(camPlaying ? "Stop" : "Play", ImVec2(80.0f, 0.0f)))
        playToggled = true;
    ImGui::EndDisabled();
    if (s.camPathCount < 2)
        ImGui::SetItemTooltip("Add at least 2 points to play a flythrough.");

    ImGui::End();

    // Frame-time graph: a separate bar pinned to the bottom, full width.
    // Hand-drawn so it behaves like a real graph: FIXED Y axis (0 at the
    // bottom, s.frameGraphMaxMs at the top) with ms labels on the left. A fixed
    // scale means samples keep their height as they scroll — no rescaling jump.
    if (s.showFrameGraph) {
        const ImGuiViewport* gvp = ImGui::GetMainViewport();
        const float graphH = 120.0f;
        ImGui::SetNextWindowPos(ImVec2(gvp->WorkPos.x,
                                       gvp->WorkPos.y + gvp->WorkSize.y - graphH),
                                ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(gvp->WorkSize.x, graphH), ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6, 6));
        ImGui::Begin("Frame Time", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoBringToFrontOnFocus);

        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImVec2 avail = ImGui::GetContentRegionAvail();

        const float leftPad = 38.0f; // room for axis labels (up to 3 digits)
        const float titleH  = 16.0f; // room for the title above the plot
        ImVec2 a{p0.x + leftPad, p0.y + titleH};      // plot area top-left
        ImVec2 b{p0.x + avail.x, p0.y + avail.y};     // plot area bottom-right
        float gw = b.x - a.x, gh = b.y - a.y;
        float maxMs = std::max(s.frameGraphMaxMs, 1.0f);

        dl->AddRectFilled(p0, b, IM_COL32(8, 8, 12, 200));

        // Title, top-left above the plot.
        dl->AddText(ImVec2(a.x, p0.y), IM_COL32(200, 200, 210, 255), "frame time (ms)");

        // Horizontal gridlines + right-aligned ms labels (0 at bottom, up).
        int step = std::max(1, (int)std::ceil(maxMs / 6.0f));
        for (int ms = 0; ms <= (int)maxMs; ms += step) {
            float y = b.y - (ms / maxMs) * gh;
            dl->AddLine(ImVec2(a.x, y), ImVec2(b.x, y), IM_COL32(60, 60, 70, 120));
            char lbl[8];
            snprintf(lbl, sizeof(lbl), "%d", ms);
            ImVec2 ts = ImGui::CalcTextSize(lbl);
            dl->AddText(ImVec2(a.x - 6.0f - ts.x, y - ts.y * 0.5f),
                        IM_COL32(170, 170, 180, 255), lbl);
        }

        // The line itself (oldest sample on the left, newest on the right).
        ImU32 lineCol = IM_COL32(90, 170, 240, 255);
        for (int i = 1; i < kHistory; i++) {
            float v0 = m_msHistory[(m_msHead + i - 1) % kHistory];
            float v1 = m_msHistory[(m_msHead + i) % kHistory];
            float x0 = a.x + (float)(i - 1) / (kHistory - 1) * gw;
            float x1 = a.x + (float)i / (kHistory - 1) * gw;
            float y0 = b.y - std::min(v0 / maxMs, 1.0f) * gh;
            float y1 = b.y - std::min(v1 / maxMs, 1.0f) * gh;
            dl->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), lineCol, 1.5f);
        }

        ImGui::End();
        ImGui::PopStyleVar();
    }
}

void Ui::render(VkCommandBuffer cmd) {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void Ui::destroy() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}
