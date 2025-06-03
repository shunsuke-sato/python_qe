using Printf
using LinearAlgebra

function list_of_orbitals(orb_name, orb_occ)
    num_orb = length(orb_name)
    l_ang_mom = zeros(Float64, num_orb)
    occupation = zeros(Float64, num_orb)
    num_node = zeros(Int, num_orb)

    num_node_s, num_node_p, num_node_d, num_node_f = 0, 0, 0, 0

    for i in 1:num_orb
        occupation[i] = orb_occ[i]
        if occursin("s", lowercase(orb_name[i]))
            l_ang_mom[i] = 0.0
            num_node[i] = num_node_s
            num_node_s += 1
        elseif occursin("p", lowercase(orb_name[i]))
            l_ang_mom[i] = 1.0
            num_node[i] = num_node_p
            num_node_p += 1
        elseif occursin("d", lowercase(orb_name[i]))
            l_ang_mom[i] = 2.0
            num_node[i] = num_node_d
            num_node_d += 1
        elseif occursin("f", lowercase(orb_name[i]))
            l_ang_mom[i] = 3.0
            num_node[i] = num_node_f
            num_node_f += 1
        end
    end

    return num_orb, l_ang_mom, occupation, num_node
end

function calc_density(num_orb, phi, occupation)
    rho = zeros(size(phi, 1))
    for i in 1:num_orb
        rho .+= occupation[i] * phi[:, i].^2
    end
    return rho
end

function calc_potential(rho, rj, dr, num_grid)
    vh = zeros(num_grid)
    for i in 1:num_grid
        v1 = sum(rj[1:i-1].^2 .* rho[1:i-1]) * dr + 0.5 * rj[i]^2 * rho[i] * dr
        v1 = i == 1 ? 0.0 : 4π * v1 / rj[i]

        v2 = 0.5 * rj[i] * rho[i] * dr + sum(rj[i+1:end] .* rho[i+1:end]) * dr
        v2 = 4π * v2
        vh[i] = v1 + v2
    end
    vxc = -((3.0 / π) * rho).^(1.0 / 3.0)
    vhxc = vh + vxc
    return vhxc, vh, vxc
end

function get_wavefunction(rj, dr, num_grid, vhxc, zval, energy, l_ang)
    chi = zeros(num_grid)
    num_node = 0
    chi[2] = dr / zval
    factor = 2 * dr^2

    for j in 2:num_grid-1
        potential = -zval / rj[j] + vhxc[j] + 0.5 * l_ang * (l_ang + 1.0) / rj[j]^2
        chi[j+1] = 2 * chi[j] - chi[j-1] - factor * (energy - potential) * chi[j]
        if chi[j+1] * chi[j] < 0
            num_node += 1
        end
    end
    return chi, num_node
end

function shooting_method(rj, dr, num_grid, vhxc, zval, l_ang_mom_in, num_node_in)
    ene_max = 0.1 * zval^2
    ene_min = -0.6 * zval^2

    for _ in 1:100
        ene_t = 0.5 * (ene_max + ene_min)
        chi, num_node_t = get_wavefunction(rj, dr, num_grid, vhxc, zval, ene_t, l_ang_mom_in)
        if num_node_t >= num_node_in + 1
            ene_max = ene_t
        else
            ene_min = ene_t
        end
        if ene_max - ene_min < 1e-6
            break
        end
    end

    ene_t = ene_max
    chi, _ = get_wavefunction(rj, dr, num_grid, vhxc, zval, ene_t, l_ang_mom_in)

    num_node_t = 0
    for j in 2:num_grid-1
        if chi[j+1] * chi[j] < 0
            num_node_t += 1
        end
        if num_node_t == num_node_in + 1
            chi[j+1:end] .= 0.0
            break
        end
    end

    norm = sum(chi.^2) * dr
    chi ./= sqrt(norm)
    phi_t = zeros(num_grid)
    phi_t[2:end-1] .= chi[2:end-1] ./ rj[2:end-1]
    phi_t[1] = 2 * phi_t[2] - phi_t[3]
    phi_t ./= sqrt(4π)

    return phi_t, ene_t
end

function calc_wavefunction(rj, dr, num_grid, vhxc, zval, num_orb, l_ang_mom, num_node)
    phi = zeros(num_grid, num_orb)
    epsilon_s = zeros(num_orb)
    for i in 1:num_orb
        phi[:, i], epsilon_s[i] = shooting_method(rj, dr, num_grid, vhxc, zval, l_ang_mom[i], num_node[i])
    end
    return phi, epsilon_s
end

function calc_total_energy(rj, dr, num_grid, phi, epsilon_s, rho, num_orb, vhxc, vh, occupation)
    total_energy = sum(occupation .* epsilon_s)
    total_energy -= 4π * sum(rho .* vhxc .* rj.^2) * dr
    total_energy += 0.5 * 4π * sum(rho .* vh .* rj.^2) * dr
    total_energy -= 4π * (3.0/4.0) * (3.0/π)^(1/3) * sum(rho.^(4/3) .* rj.^2) * dr
    return total_energy
end

function calc_atom_gs(orb_name, orb_occ, nscf, rmax, dr, zval)
    num_orb, l_ang_mom, occupation, num_node = list_of_orbitals(orb_name, orb_occ)
    num_grid = Int(rmax / dr) + 1
    rj = range(0.0, rmax, length=num_grid)
    phi = zeros(num_grid, num_orb)
    epsilon_s = zeros(num_orb)
    rho_old = zeros(num_grid)
    update_rate = 0.5
    total_energy = 0.0

    for iscf in 1:nscf
        rho = calc_density(num_orb, phi, occupation)
        rho = update_rate * rho + (1.0 - update_rate) * rho_old
        rho_old = rho
        vhxc, vh, vxc = calc_potential(rho, rj, dr, num_grid)
        phi, epsilon_s = calc_wavefunction(rj, dr, num_grid, vhxc, zval, num_orb, l_ang_mom, num_node)
        total_energy = calc_total_energy(rj, dr, num_grid, phi, epsilon_s, rho, num_orb, vhxc, vh, occupation)
        @printf("iscf = %d, energy = %f Hartree\n", iscf, total_energy)
    end

    return num_orb, rj, phi, epsilon_s, total_energy
end

# Parameters
zval = 10.0
nscf = 30
rmax = 20.0
dr = 0.005

orb_name = ["1s", "2s", "2p"]
orb_occ = [2.0, 2.0, 6.0]

num_orb, rj, phi, epsilon_s, total_energy = calc_atom_gs(orb_name, orb_occ, nscf, rmax, dr, zval)

ev = 27.2114
println("Total energy = $total_energy Hartree = ", total_energy * ev, " eV")
println("Single particle energies:")
for i in 1:num_orb
    println("$(orb_name[i]) : $(epsilon_s[i]) Hartree = ", epsilon_s[i]*ev, " eV")
end
