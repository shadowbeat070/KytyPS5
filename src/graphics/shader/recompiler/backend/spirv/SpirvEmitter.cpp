#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

using ShaderError::Fail;

namespace {

bool ValidateNativeProgram(const IR::Program& program, std::string* error) {
	using Kind                                             = IR::DescriptorBindingKind;
	constexpr auto                               KindCount = static_cast<size_t>(Kind::Count);
	std::array<std::vector<uint32_t>, KindCount> expected;
	std::array<bool, KindCount>                  present {};
	const auto                                   Dense = [](size_t size) {
		std::vector<uint32_t> values(size);
		for (uint32_t i = 0; i < values.size(); i++) {
			values[i] = i;
		}
		return values;
	};
	auto Expect = [&](Kind kind, std::vector<uint32_t> resources = {}) {
		const auto index = static_cast<size_t>(kind);
		present[index]   = true;
		expected[index]  = std::move(resources);
	};
	if (!program.info.buffers.empty()) {
		Expect(Kind::Buffers, Dense(program.info.buffers.size()));
	}
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto kind = IR::DescriptorBindingForImage(program.info.images[i]);
		if (!kind.has_value()) {
			return Fail(error, "native shader plan has an invalid image class");
		}
		present[static_cast<size_t>(*kind)] = true;
		const auto dynamic = program.info.images[i].mip_mode == IR::ImageMipMode::DynamicStorage;
		const auto count   = dynamic ? program.info.images[i].mip_count : 1u;
		if (count == 0u || (!dynamic && program.info.images[i].mip_count != 1u)) {
			return Fail(error, "native shader plan has an invalid image mip descriptor count");
		}
		expected[static_cast<size_t>(*kind)].insert(expected[static_cast<size_t>(*kind)].end(),
		                                            count, i);
	}
	if (!program.info.samplers.empty()) {
		Expect(Kind::Samplers, Dense(program.info.samplers.size()));
	}
	bool uses_gds = false;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (IR::SharedAccessOf(inst.GetOpcode()) == IR::SharedAccess::None) {
				continue;
			}
			const auto index = inst.Flags<IR::MemoryFlags>().index;
			if (index >= program.memory_info.size()) {
				return Fail(error, "shared operation has invalid memory metadata");
			}
			const auto kind = program.memory_info[index].kind;
			if (kind != IR::ResourceKind::Lds && kind != IR::ResourceKind::Gds) {
				return Fail(error, "shared operation has invalid resource kind");
			}
			uses_gds |= kind == IR::ResourceKind::Gds;
		}
	}
	if (uses_gds) {
		Expect(Kind::Gds);
	}
	if (program.info.uses_dma) {
		Expect(Kind::BdaPagetable);
		Expect(Kind::FaultBuffer);
	}
	const bool uses_flattened_runtime =
	    !program.srt_reads.empty() ||
	     std::ranges::any_of(program.info.images, [](const IR::ImageResource& image) {
		     return image.indirect_mapping_capacity != 0u;
	     });
	if (uses_flattened_runtime) {
		Expect(Kind::FlattenedSrt);
	}
	if (std::ranges::any_of(program.blocks, [](const IR::Block* block) {
		    return std::ranges::any_of(*block, [](const auto& inst) {
			    return inst.GetOpcode() == IR::ValueOpcode::ReadMeshIndex;
		    });
	    })) {
		Expect(Kind::MeshIndices);
	}
	if (program.bindings.ShaderDataDwords() != 0 && program.bindings.push_constant_size == 0) {
		Expect(Kind::UserData);
	}

	std::array<bool, KindCount> seen {};
	for (const auto& binding: program.bindings.descriptors) {
		const auto kind = static_cast<size_t>(binding.kind);
		if (kind >= KindCount || seen[kind] || !present[kind] ||
		    binding.resources != expected[kind]) {
			return Fail(error, "native descriptor groups do not match shader topology");
		}
		seen[kind] = true;
	}
	for (size_t i = 0; i < KindCount; i++) {
		if (present[i] != seen[i]) {
			return Fail(error, "native shader plan is missing a required descriptor group");
		}
	}
	const auto has_user_storage = present[static_cast<size_t>(Kind::UserData)];
	if (program.bindings.push_constant_offset % sizeof(uint32_t) != 0 ||
	    program.bindings.push_constant_offset + program.bindings.push_constant_size >
	        IR::NativePushConstantSize ||
	    program.bindings.memory_offset_dword != program.bindings.user_data_registers.size() ||
	    program.bindings.memory_offset_count != program.info.buffers.size() ||
	    (!has_user_storage &&
	     program.bindings.push_constant_size != program.bindings.ShaderDataDwords() * 4u) ||
	    (has_user_storage && program.bindings.push_constant_size != 0) ||
	    !std::is_sorted(program.bindings.user_data_registers.begin(),
	                    program.bindings.user_data_registers.end()) ||
	    std::adjacent_find(program.bindings.user_data_registers.begin(),
	                       program.bindings.user_data_registers.end()) !=
	        program.bindings.user_data_registers.end()) {
		return Fail(error, "native user-data layout is inconsistent");
	}

	const auto planning_only_handle = [&](const IR::Inst& handle) {
		return !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       const auto op = use.user->GetOpcode();
			       if (op != IR::ValueOpcode::LoadAddressU32 &&
			           op != IR::ValueOpcode::ReadConstBuffer) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].planning_only;
		       });
	};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto dense = inst.Flags<uint32_t>();
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::GetBufferResource:
					if (planning_only_handle(inst)) {
						break;
					}
					if (dense >= program.info.buffers.size()) {
						return Fail(error, "typed buffer handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetAddressResource:
					if (planning_only_handle(inst)) {
						break;
					}
					if (inst.NumArgs() != 2 || !program.info.uses_dma) {
						return Fail(error, "typed address handle has invalid DMA metadata");
					}
					break;
				case IR::ValueOpcode::GetScratchResource:
					if (inst.NumArgs() != 0 || program.scratch_dwords == 0) {
						return Fail(error, "typed scratch handle has invalid shader metadata");
					}
					break;
				case IR::ValueOpcode::GetImageResource:
					if (dense >= program.info.images.size()) {
						return Fail(error, "typed image handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetSamplerResource:
					if (dense >= program.info.samplers.size()) {
						return Fail(error, "typed sampler handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::ReadConst: {
					const auto slot = inst.Arg(1).Resolve();
					if (!slot.IsImmediate() || slot.GetType() != IR::Type::U32 ||
					    slot.U32() >= program.srt_reads.size()) {
						return Fail(error, "flattened SRT read has an invalid dense slot");
					}
					break;
				}
				default: break;
			}
		}
	}
	return true;
}

// Bounds a shared-memory byte address so a stage emulating LDS in per-invocation storage allocates
// what the program can reach, not the conservative maximum. Empty when not provably bounded.
std::optional<uint32_t> SharedAddressUpperBound(IR::Value value, uint32_t wave_size,
                                                uint32_t depth) {
	constexpr uint32_t MaxDepth = 32;
	if (depth >= MaxDepth) {
		return std::nullopt;
	}
	value = value.Resolve();
	if (value.IsImmediate()) {
		return value.GetType() == IR::Type::U32 ? std::optional<uint32_t>(value.U32())
		                                        : std::nullopt;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return std::nullopt;
	}
	const auto Operand = [&](size_t index) {
		return SharedAddressUpperBound(inst->Arg(index), wave_size, depth + 1);
	};
	const auto Immediate = [&](size_t index) -> std::optional<uint32_t> {
		const auto arg = inst->Arg(index).Resolve();
		return arg.IsImmediate() && arg.GetType() == IR::Type::U32
		           ? std::optional<uint32_t>(arg.U32())
		           : std::nullopt;
	};
	switch (inst->GetOpcode()) {
		case IR::ValueOpcode::LaneId: return wave_size == 0 ? 0u : wave_size - 1u;
		case IR::ValueOpcode::IAdd32: {
			const auto lhs = Operand(0);
			const auto rhs = Operand(1);
			if (!lhs.has_value() || !rhs.has_value()) {
				return std::nullopt;
			}
			const auto sum = static_cast<uint64_t>(*lhs) + *rhs;
			return sum > UINT32_MAX ? std::nullopt : std::optional<uint32_t>(sum);
		}
		case IR::ValueOpcode::BitwiseAnd32: {
			// A constant mask bounds the result no matter what the other operand holds.
			const auto lhs_mask = Immediate(0);
			const auto rhs_mask = Immediate(1);
			auto       bound    = lhs_mask.has_value() ? lhs_mask : rhs_mask;
			if (rhs_mask.has_value() && (!bound.has_value() || *rhs_mask < *bound)) {
				bound = rhs_mask;
			}
			const auto lhs = Operand(0);
			const auto rhs = Operand(1);
			for (const auto& candidate: {lhs, rhs}) {
				if (candidate.has_value() && (!bound.has_value() || *candidate < *bound)) {
					bound = candidate;
				}
			}
			return bound;
		}
		case IR::ValueOpcode::ShiftLeftLogical32: {
			const auto lhs   = Operand(0);
			const auto shift = Immediate(1);
			if (!lhs.has_value() || !shift.has_value() || *shift >= 32u) {
				return std::nullopt;
			}
			const auto shifted = static_cast<uint64_t>(*lhs) << *shift;
			return shifted > UINT32_MAX ? std::nullopt : std::optional<uint32_t>(shifted);
		}
		case IR::ValueOpcode::ShiftRightLogical32: {
			const auto lhs   = Operand(0);
			const auto shift = Immediate(1);
			if (!lhs.has_value() || !shift.has_value() || *shift >= 32u) {
				return std::nullopt;
			}
			return static_cast<uint32_t>(*lhs >> *shift);
		}
		default: return std::nullopt;
	}
}

} // namespace

bool AnalyzeProgramRequirements(IR::Program& program, std::string* error) {
	program.spirv_requirements.reset();
	IR::SpirvRequirements requirements {};
	const auto MarkBallot = [&] { requirements.subgroup_ballot = true; };
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto address_access = IR::AddressOpcodeInfoOf(inst.GetOpcode()).access;
			if (address_access != IR::AddressAccess::None) {
				const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					return Fail(error, "address operation has invalid memory metadata");
				}
				if (program.memory_info[memory_index].kind == IR::ResourceKind::Scratch) {
					if (program.scratch_dwords == 0) {
						return Fail(error, "scratch operation has no per-thread storage");
					}
					requirements.function_scratch = true;
				} else if (address_access == IR::AddressAccess::Write) {
					return Fail(error, "writable FLAT/GLOBAL addresses require GPU ownership tracking");
				}
			}
			if (IR::BufferAccessOf(inst.GetOpcode()) != IR::BufferAccess::None) {
				const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					return Fail(error, "buffer operation has invalid memory metadata");
				}
				const auto& memory = program.memory_info[memory_index];
				if (memory.kind == IR::ResourceKind::Buffer) {
					if (memory.resource >= program.info.buffers.size()) {
						return Fail(error, "buffer operation has invalid resource metadata");
					}
					if ((program.info.buffers[memory.resource].packed_stride & (1u << 20u)) != 0u) {
						if (program.stage != ShaderType::Compute) {
							return Fail(error, "buffer ADD_TID is only valid for compute shaders");
						}
						requirements.subgroup_local_invocation_id = true;
					}
				}
			}
			const auto shared_access = IR::SharedAccessOf(inst.GetOpcode());
			if (shared_access != IR::SharedAccess::None) {
				const auto index = inst.Flags<IR::MemoryFlags>().index;
				if (index >= program.memory_info.size()) {
					return Fail(error, "shared operation has invalid memory metadata");
				}
				const auto kind = program.memory_info[index].kind;
				if (kind != IR::ResourceKind::Lds && kind != IR::ResourceKind::Gds) {
					return Fail(error, "shared operation has invalid resource kind");
				}
				if (program.stage != ShaderType::Compute && program.stage != ShaderType::Mesh &&
				    kind == IR::ResourceKind::Lds) {
					if (!requirements.function_lds) {
						requirements.function_lds        = true;
						requirements.function_lds_dwords = 1;
					}
					const auto&             mem = program.memory_info[index];
					std::optional<uint32_t> bound;
					if (shared_access != IR::SharedAccess::Append &&
					    shared_access != IR::SharedAccess::Consume) {
						// Outside compute, LaneId lowers to the host subgroup invocation id,
						// which Vulkan allows to exceed the guest wave size (up to 128).
						bound = SharedAddressUpperBound(
						    inst.Arg(0), std::max<uint32_t>(program.wave_size, 128u), 0);
					}
					if (!bound.has_value() || requirements.function_lds_dwords == 0) {
						requirements.function_lds_dwords = 0;
					} else {
						const auto last = (static_cast<uint64_t>(*bound) + mem.offset) / 4u +
						                  std::max<uint32_t>(mem.data_dwords, 1u);
						requirements.function_lds_dwords =
						    last > UINT32_MAX
						        ? 0u
						        : std::max<uint32_t>(requirements.function_lds_dwords,
						                             static_cast<uint32_t>(last));
					}
				}
				if (shared_access == IR::SharedAccess::Append ||
				    shared_access == IR::SharedAccess::Consume) {
					MarkBallot();
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
				}
			}
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::Ballot: MarkBallot(); break;
				case IR::ValueOpcode::DppMoveU32:
				case IR::ValueOpcode::ReadFirstLane:
				case IR::ValueOpcode::ReadLane: {
					MarkBallot();
					requirements.subgroup_shuffle = true;
					if (inst.GetOpcode() == IR::ValueOpcode::DppMoveU32) {
						requirements.subgroup_local_invocation_id = true;
					}
					break;
				}
				case IR::ValueOpcode::DppUpdateU32:
				case IR::ValueOpcode::WqmMask:
				case IR::ValueOpcode::WriteLane: {
					MarkBallot();
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::Permlane16U32: {
					MarkBallot();
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::SwizzleU32: {
					MarkBallot();
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::LaneId:
					requirements.subgroup_local_invocation_id = true;
					break;
				case IR::ValueOpcode::ImageQueryLod: requirements.compute_derivatives = true; break;
				case IR::ValueOpcode::ImageGatherRaw:
					requirements.image_gather_extended = true;
					break;
				case IR::ValueOpcode::SetAttribute: {
					const auto index = inst.Flags<IR::ExportFlags>().index;
					if (index >= program.export_info.size()) {
						return Fail(error, "attribute export has invalid metadata");
					}
					if (program.stage == ShaderType::Pixel &&
					    program.export_info[index].vm) {
						requirements.pixel_valid_mask = true;
					}
					break;
				}
				default: break;
			}
		}
	}
	program.spirv_requirements.emplace(requirements);
	return true;
}

bool EmitProgram(const IR::Program& program, const IR::ResourceSnapshot& resources,
                 ShaderStageInputInfo input_info, std::vector<uint32_t>& spirv,
                 std::string* error) {
	using namespace Emitter;

	if (program.stage != ShaderType::Compute && program.stage != ShaderType::Vertex &&
	    program.stage != ShaderType::Pixel && program.stage != ShaderType::Mesh) {
		SetError(error,
		         "binary SPIR-V emitter supports compute, vertex, pixel, and mesh shaders");
		return false;
	}
	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete || !program.binding_layout_complete ||
	    !program.spirv_requirements.has_value()) {
		SetError(error, "SPIR-V emitter requires a fully planned native shader program");
		return false;
	}
	if (!IR::ValidateResourceSnapshot(program, resources, error)) {
		return false;
	}
	if (!IR::ValidateResourceSpecialization(program, resources, error)) {
		return false;
	}
	if (!ValidateNativeProgram(program, error)) {
		return false;
	}
	if (!IR::ValidateProgram(program, true, error)) {
		return false;
	}
	EmitterState state(program, resources, input_info);
	state.stage     = program.stage;
	state.wave_size = program.wave_size;
	state.logical_wave64 =
	    program.stage == ShaderType::Mesh && program.wave_size == LogicalWave64Lanes;
	state.inputs.reserve(program.info.inputs.size());
	state.outputs.reserve(program.info.outputs.size());
	state.interface_variables.reserve(program.info.inputs.size() + program.info.outputs.size());
	CopyProgramInputsAndOutputs(state, program);
	AllocateInputVariables(state);
	AllocateOutputVariables(state);
	DefineModule(state);
	if (!EmitProgram(state, program, error)) {
		return false;
	}

	auto binary = state.builder.Build();
	if (binary.empty()) {
		SetError(error, "SPIR-V builder returned an empty module");
		return false;
	}
	spirv = std::move(binary);
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
