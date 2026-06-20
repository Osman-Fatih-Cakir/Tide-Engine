#include "ui.h"
#include "vk_engine.h"
#include "settings.h"

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

    // Pin the panel to the top-right corner; fixed size, not movable/resizable.
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x - 10.0f,
                                   vp->WorkPos.y + 10.0f),
                            ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(240.0f, 400.0f), ImGuiCond_Always);

    ImGui::Begin("Tide Engine", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    ImGui::Text("Avg: %.2f ms  (%.0f FPS)", m_dispAvgMs, avgFps);
    ImGui::SetItemTooltip("Average frame time over the last second.");
    ImGui::Text("Max: %.2f ms  (%.0f FPS)", m_dispMaxMs, maxFps);
    ImGui::SetItemTooltip("Slowest frame in the last second (worst-case).");

    ImGui::SeparatorText("Sun");
    ImGui::PushItemWidth(150.0f); // fixed slider width
    ImGui::SliderFloat("Azimuth",   &s.sunAzimuthDeg,   0.0f, 360.0f);
    ImGui::SliderFloat("Elevation", &s.sunElevationDeg, -10.0f, 90.0f);
    ImGui::SliderFloat("Ambient",   &s.ambient,         0.0f, 1.0f);
    ImGui::PopItemWidth();
    ImGui::End();
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
