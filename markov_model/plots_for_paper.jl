
var6 = var(r6)
corr6 = [mean(r6[i:end] .* r6[1:end-i+1])/var6 - mean(r6[i:end]) * mean(r6[1:end-i+1])/var6  for i in 1:500]
indlast = 200
cortimes = collect(0:indlast-1) .* (dt * X * 25 / 86400)
figcor = Figure()
axcor = Axis(figcor[1,1], xlabel="time (days)", ylabel="correlation")
lines!(axcor, cortimes , corr6[1:indlast])