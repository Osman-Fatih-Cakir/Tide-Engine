#include "ui.h"
#include "vk_engine.h"
#include "settings.h"

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

void Ui::buildPanel(Settings& s, float dt) {
    // Accumulate stats; refresh the displayed numbers once per second.
    m_acc += dt;
    m_frames++;
    m_maxDt = std::max(m_maxDt, dt);
    if (m_acc >= 1.0f) {
        m_dispAvgMs = (m_acc / m_frames) * 1000.0f;
        m_dispMaxMs = m_maxDt * 1000.0f;
        m_acc = 0.0f;
        m_frames = 0;
        m_maxDt = 0.0f;
    }
    float avgFps = m_dispAvgMs > 0.0f ? 1000.0f / m_dispAvgMs : 0.0f;
    float maxFps = m_dispMaxMs > 0.0f ? 1000.0f / m_dispMaxMs : 0.0f;

    // Feed the plot at a fixed rate (s.frameGraphHz samples/sec): accumulate,
    // emit one averaged point per window so the graph scrolls at a readable,
    // FPS-independent pace.
    int hz = s.frameGraphHz > 0 ? s.frameGraphHz : 1;
    float sampleInterval = 1.0f / (float)hz;
    m_plotAcc += dt;
    m_plotFrames++;
    if (m_plotAcc >= sampleInterval) {
        m_msHistory[m_msHead] = (m_plotAcc / m_plotFrames) * 1000.0f;
        m_msHead = (m_msHead + 1) % kHistory;
        m_plotAcc = 0.0f;
        m_plotFrames = 0;
    }

    // Pin the panel to the top-right corner; fixed size, not movable/resizable.
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x - 10.0f,
                                   vp->WorkPos.y + 10.0f),
                            ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(240.0f, 470.0f), ImGuiCond_Always);

    ImGui::Begin("Tide Engine", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    ImGui::Text("Avg: %.2f ms  (%.0f FPS)", m_dispAvgMs, avgFps);
    ImGui::SetItemTooltip("Average frame time over the last second.");
    ImGui::Text("Max: %.2f ms  (%.0f FPS)", m_dispMaxMs, maxFps);
    ImGui::SetItemTooltip("Slowest frame in the last second (worst-case).");

    ImGui::Checkbox("VSync", &s.vsync);
    ImGui::SetItemTooltip("On: FIFO (no tearing). Off: MAILBOX/IMMEDIATE (uncapped).");
    ImGui::Checkbox("Frame graph", &s.showFrameGraph);
    ImGui::SetNextItemWidth(150.0f);
    ImGui::SliderInt("Graph Hz", &s.frameGraphHz, 1, 60);
    ImGui::SetItemTooltip("Graph samples per second.");
    ImGui::SetNextItemWidth(150.0f);
    ImGui::SliderFloat("Graph Max", &s.frameGraphMaxMs, 2.0f, 200.0f, "%.0f ms");
    ImGui::SetItemTooltip("Fixed Y-axis top of the frame-time graph.");

    ImGui::SeparatorText("Sun");
    ImGui::PushItemWidth(150.0f); // fixed slider width
    ImGui::SliderFloat("Azimuth",   &s.sunAzimuthDeg,   0.0f, 360.0f);
    ImGui::SliderFloat("Elevation", &s.sunElevationDeg, -10.0f, 90.0f);
    ImGui::SliderFloat("Intensity", &s.sunIntensity,    0.0f, 20.0f);
    ImGui::SliderFloat("Ambient",   &s.ambient,         0.0f, 1.0f);

    ImGui::SeparatorText("Shadows (RT)");
    ImGui::Checkbox("Enabled", &s.shadowsEnabled);
    ImGui::SliderFloat("Softness", &s.sunAngularSize, 0.0f, 5.0f, "%.2f deg");
    ImGui::SliderInt("Samples",    &s.shadowSamples,  1, 16);
    ImGui::Checkbox("Denoise (temporal)", &s.shadowDenoise);
    ImGui::SliderFloat("History",  &s.shadowHistAlpha, 0.02f, 1.0f, "%.2f");

    ImGui::SeparatorText("DLSS (Faz 6.5)");
    ImGui::Checkbox("DLSS", &s.dlssEnabled);
    const char* dlssModes[] = {"Performance", "Balanced", "Quality", "Ultra Performance", "DLAA"};
    ImGui::Combo("DLSS Mode", &s.dlssQuality, dlssModes, IM_ARRAYSIZE(dlssModes));
    // Live status: confirms whether DLSS is really upscaling.
    if (!s.dlssAvailable) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "Status: NOT AVAILABLE (native)");
    } else if (s.dlssActive) {
        ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1), "Status: ACTIVE  %ux%u -> %ux%u",
                           s.renderW, s.renderH, s.displayW, s.displayH);
    } else {
        ImGui::TextColored(ImVec4(1, 1, 0.4f, 1), "Status: off (native %ux%u)", s.displayW, s.displayH);
    }
    ImGui::Checkbox("Debug: motion vectors", &s.debugMotionVecs);

    ImGui::SeparatorText("Tonemap");
    ImGui::SliderFloat("Exposure",  &s.exposure,        0.1f, 5.0f);
    ImGui::PopItemWidth();
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
