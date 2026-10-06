using LidJul, GLMakie

# GLFW reports no monitor in some sandboxed or headless macOS sessions.
# Guard the display path before GLMakie queries a null primary monitor.
isempty(GLMakie.GLFW.GetMonitors()) && error(
    "No monitor is visible to GLFW. Run from a graphical desktop session, " *
    "or use CairoMakie in the examples environment to export a figure."
)

result = simulate_cavity(CavityConfig(nx=32, tf=2))
screen = display(plot_cavity(result.state))
isinteractive() || wait(screen)
